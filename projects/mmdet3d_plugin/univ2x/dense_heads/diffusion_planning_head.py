import math
import os
import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS

from .planning_head import PlanningHeadSingleMode

try:
    from diffusers.schedulers import DDIMScheduler
except ImportError:
    DDIMScheduler = None


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def bias_init_with_prob(prior_prob):
    return float(-np.log((1 - prior_prob) / prior_prob))


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.floor(dim_t / 2) / half_hidden_dim)

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def py_sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    return loss


class _FallbackDDIMScheduler(object):
    """Small DDIM-compatible fallback for environments without diffusers."""

    def __init__(self, num_train_timesteps=1000, **kwargs):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None

    def add_noise(self, original_samples, noise, timesteps):
        while timesteps.dim() < original_samples.dim():
            timesteps = timesteps.unsqueeze(-1)
        scale = timesteps.float() / float(max(self.num_train_timesteps - 1, 1))
        return original_samples * (1.0 - scale) + noise * scale

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1, device=device)

    def step(self, model_output, timestep, sample):
        return SimpleNamespace(prev_sample=model_output)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class UniV2XGridSampleCrossBEVAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_points=10, bev_range=(-51.2, -51.2, 51.2, 51.2)):
        super(UniV2XGridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.bev_range = bev_range
        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.value_proj = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)
        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature):
        bs, num_queries, num_points, _ = traj_points.shape
        x_min, y_min, x_max, y_max = self.bev_range

        grid = traj_points.new_zeros(bs, num_queries, num_points, 2)
        grid[..., 0] = 2.0 * (traj_points[..., 0] - x_min) / (x_max - x_min) - 1.0
        grid[..., 1] = -(2.0 * (traj_points[..., 1] - y_min) / (y_max - y_min) - 1.0)
        grid = torch.clamp(grid, min=-1.5, max=1.5)

        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        sampled_features = F.grid_sample(
            value,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        out = (attention_weights.unsqueeze(1) * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()
        out = self.output_proj(out)
        return self.dropout(out) + queries


class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, ego_fut_ts=10, ego_fut_mode=20):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.init_weight()

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature):
        bs, ego_fut_mode, _ = traj_feature.shape
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 2)
        return plan_reg, plan_cls


class ModulationLayer(nn.Module):
    def __init__(self, embed_dims, condition_dims):
        super(ModulationLayer, self).__init__()
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )

    def forward(self, traj_feature, time_embed):
        scale, shift = self.scale_shift_mlp(time_embed).chunk(2, dim=-1)
        return traj_feature * (1 + scale) + shift


class DiffusionPlanningDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, num_points, bev_range):
        super(DiffusionPlanningDecoderLayer, self).__init__()
        self.cross_bev_attention = UniV2XGridSampleCrossBEVAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
            bev_range=bev_range,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            embed_dims,
            num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            embed_dims,
            num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(0.1)
        self.time_modulation = ModulationLayer(embed_dims, embed_dims)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=embed_dims,
            ego_fut_ts=num_points,
        )

    def forward(self, traj_feature, noisy_traj_points, bev_feature, agents_query, ego_query, time_embed):
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature)
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]
        )
        traj_feature = self.norm1(traj_feature)
        traj_feature = traj_feature + self.dropout(
            self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]
        )
        traj_feature = self.norm2(traj_feature)
        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature = self.time_modulation(traj_feature, time_embed)

        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg = poses_reg + noisy_traj_points
        return poses_reg, poses_cls


class DiffusionPlanningDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(DiffusionPlanningDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, traj_feature, noisy_traj_points, bev_feature, agents_query, ego_query, time_embed):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for layer in self.layers:
            poses_reg, poses_cls = layer(
                traj_feature,
                traj_points,
                bev_feature,
                agents_query,
                ego_query,
                time_embed,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg.detach()
        return poses_reg_list, poses_cls_list


@HEADS.register_module()
class DiffusionPlanningHead(PlanningHeadSingleMode):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=10,
                 loss_planning=None,
                 loss_collision=None,
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0,
                    alpha_collision=5.0,
                 ),
                 with_adapter=False,
                 occ_n_future_only_occ=4,
                 num_modes=20,
                 num_heads=8,
                 num_decoder_layers=2,
                 num_train_timesteps=1000,
                 train_noise_timesteps=50,
                 inference_steps=2,
                 plan_anchor_path=None,
                 xy_norm_scale=(60.0, 60.0),
                 xy_norm_bounds=None,
                 bev_range=(-51.2, -51.2, 51.2, 51.2),
                 num_agent_queries=30,
                 num_command_classes=3,
                 ego_status_dim=4,
                 loss_diffusion_cls_weight=10.0,
                 loss_diffusion_reg_weight=8.0,
                 train_aux_planning_losses=True):
        super(DiffusionPlanningHead, self).__init__(
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=embed_dims,
            planning_steps=planning_steps,
            loss_planning=loss_planning,
            loss_collision=loss_collision,
            planning_eval=planning_eval,
            use_col_optim=use_col_optim,
            col_optim_args=col_optim_args,
            with_adapter=with_adapter,
            occ_n_future_only_occ=occ_n_future_only_occ,
        )
        self.num_modes = num_modes
        self.embed_dims = embed_dims
        self.inference_steps = inference_steps
        self.train_noise_timesteps = train_noise_timesteps
        self.xy_norm_scale = xy_norm_scale
        self.bev_range = bev_range
        self.num_agent_queries = num_agent_queries
        self.num_command_classes = num_command_classes
        self.ego_status_dim = ego_status_dim
        self.loss_diffusion_cls_weight = loss_diffusion_cls_weight
        self.loss_diffusion_reg_weight = loss_diffusion_reg_weight
        self.train_aux_planning_losses = train_aux_planning_losses

        scheduler_cls = DDIMScheduler if DDIMScheduler is not None else _FallbackDDIMScheduler
        self.diffusion_scheduler = scheduler_cls(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='scaled_linear',
            prediction_type='sample',
        )

        self.agent_query_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        self.status_encoder = nn.Sequential(
            nn.Linear(num_command_classes + ego_status_dim, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        self.ego_query_fuser = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, planning_steps * embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.Mish(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        decoder_layer = DiffusionPlanningDecoderLayer(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=planning_steps,
            bev_range=bev_range,
        )
        self.diff_decoder = DiffusionPlanningDecoder(decoder_layer, num_decoder_layers)

        plan_anchor = self._load_or_build_plan_anchor(plan_anchor_path)
        xy_norm_min, xy_norm_max = self._resolve_xy_norm_bounds(plan_anchor, xy_norm_bounds)
        self.register_buffer('plan_anchor', plan_anchor)
        self.register_buffer('xy_norm_min', xy_norm_min, persistent=False)
        self.register_buffer('xy_norm_max', xy_norm_max, persistent=False)

    def _load_or_build_plan_anchor(self, plan_anchor_path):
        if plan_anchor_path and os.path.exists(plan_anchor_path):
            plan_anchor = np.load(plan_anchor_path).astype(np.float32)
            if plan_anchor.shape[:2] != (self.num_modes, self.planning_steps):
                raise ValueError(
                    'plan_anchor_path must contain shape '
                    '({}, {}, 2), got {}'.format(self.num_modes, self.planning_steps, plan_anchor.shape)
                )
            return torch.from_numpy(plan_anchor[..., :2])

        ts = np.arange(1, self.planning_steps + 1, dtype=np.float32) * 0.5
        speeds = np.linspace(0.5, 7.0, 5, dtype=np.float32)
        lateral_offsets = np.array([-4.0, -1.5, 1.5, 4.0], dtype=np.float32)
        anchors = []
        for speed in speeds:
            for lateral in lateral_offsets:
                progress = ts / ts[-1]
                x = speed * ts
                y = lateral * progress * progress
                anchors.append(np.stack([x, y], axis=-1))
        anchors = np.stack(anchors, axis=0)[:self.num_modes]
        return torch.from_numpy(anchors.astype(np.float32))

    def _resolve_xy_norm_bounds(self, plan_anchor, xy_norm_bounds=None):
        if xy_norm_bounds is not None:
            bounds = torch.as_tensor(xy_norm_bounds, dtype=plan_anchor.dtype)
            if bounds.numel() == 4:
                bounds = bounds.view(2, 2)
            if bounds.shape != (2, 2):
                raise ValueError('xy_norm_bounds must be (2, 2) or 4 values, got {}'.format(tuple(bounds.shape)))
            return bounds[0].clone(), bounds[1].clone()

        xy_min = plan_anchor.amin(dim=(0, 1))
        xy_max = plan_anchor.amax(dim=(0, 1))
        span = torch.clamp(xy_max - xy_min, min=1.0)
        margin = torch.maximum(span * 0.10, plan_anchor.new_tensor([2.0, 2.0]))
        return xy_min - margin, xy_max + margin

    def _norm_xy(self, xy):
        xy_min = self.xy_norm_min.to(device=xy.device, dtype=xy.dtype)
        xy_max = self.xy_norm_max.to(device=xy.device, dtype=xy.dtype)
        xy = 2.0 * (xy - xy_min) / (xy_max - xy_min) - 1.0
        return torch.clamp(xy, min=-1.0, max=1.0)

    def _denorm_xy(self, xy):
        xy_min = self.xy_norm_min.to(device=xy.device, dtype=xy.dtype)
        xy_max = self.xy_norm_max.to(device=xy.device, dtype=xy.dtype)
        return (xy + 1.0) * 0.5 * (xy_max - xy_min) + xy_min

    def _format_command(self, command, batch_size, device):
        if isinstance(command, (list, tuple)):
            command = command[0]
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=device)
        command = command.to(device=device, dtype=torch.long).reshape(-1)
        if command.numel() == 1 and batch_size > 1:
            command = command.expand(batch_size)
        command = command[:batch_size]
        return torch.clamp(command, min=0, max=self.num_command_classes - 1)

    def _format_ego_status(self, ego_status, batch_size, device, dtype):
        if isinstance(ego_status, (list, tuple)):
            ego_status = ego_status[0]
        if ego_status is None:
            return torch.zeros(batch_size, self.ego_status_dim, device=device, dtype=dtype)
        if not torch.is_tensor(ego_status):
            ego_status = torch.tensor(ego_status, device=device)
        ego_status = ego_status.to(device=device, dtype=dtype)
        if ego_status.dim() == 1:
            ego_status = ego_status[None, :]
        elif ego_status.dim() > 2:
            ego_status = ego_status.reshape(ego_status.shape[0], -1)
        if ego_status.shape[0] == 1 and batch_size > 1:
            ego_status = ego_status.expand(batch_size, -1)
        ego_status = ego_status[:batch_size]
        if ego_status.shape[-1] < self.ego_status_dim:
            pad = ego_status.new_zeros(ego_status.shape[0], self.ego_status_dim - ego_status.shape[-1])
            ego_status = torch.cat([ego_status, pad], dim=-1)
        return ego_status[:, :self.ego_status_dim]

    def _as_batched_query(self, query, batch_size=None):
        if query is None:
            return None
        if isinstance(query, (list, tuple)):
            query = query[0]
        if query.dim() == 1:
            query = query[None, None, :]
        elif query.dim() == 2:
            if batch_size is not None and query.shape[0] == batch_size:
                query = query[:, None, :]
            else:
                query = query[None, :, :]
        return query

    def _pad_agent_queries(self, agent_queries, batch_size, device, dtype):
        if agent_queries is None:
            return torch.zeros(batch_size, self.num_agent_queries, self.embed_dims, device=device, dtype=dtype)
        agent_queries = self._as_batched_query(agent_queries, batch_size=batch_size)
        agent_queries = agent_queries.to(device=device, dtype=dtype)
        if agent_queries.shape[0] == 1 and batch_size > 1:
            agent_queries = agent_queries.expand(batch_size, -1, -1)

        num_agents = agent_queries.shape[1]
        if num_agents >= self.num_agent_queries:
            return agent_queries[:, :self.num_agent_queries]

        pad = agent_queries.new_zeros(batch_size, self.num_agent_queries - num_agents, self.embed_dims)
        return torch.cat([agent_queries, pad], dim=1)

    def _build_condition(self, outs_motion, command):
        sdc_embedding = outs_motion.get('planning_sdc_embedding', None)
        if sdc_embedding is None:
            sdc_embedding = outs_motion.get('sdc_track_query', None)
        track_query_embeddings = outs_motion.get('planning_track_query_embeddings', None)
        if track_query_embeddings is None:
            track_query_embeddings = outs_motion.get('track_query', None)

        if sdc_embedding is None:
            raise KeyError('DiffusionPlanningHead requires planning_sdc_embedding or sdc_track_query')

        if isinstance(sdc_embedding, (list, tuple)):
            sdc_embedding = sdc_embedding[0]
        if sdc_embedding.dim() == 1:
            sdc_embedding = sdc_embedding[None, :]
        elif sdc_embedding.dim() == 3:
            sdc_embedding = sdc_embedding.squeeze(1)
        batch_size = sdc_embedding.shape[0]
        device = sdc_embedding.device
        dtype = sdc_embedding.dtype

        command = self._format_command(command, batch_size, device)
        command_onehot = F.one_hot(command, num_classes=self.num_command_classes).to(dtype=dtype)
        ego_status = self._format_ego_status(
            outs_motion.get('planning_ego_status', None),
            batch_size,
            device,
            dtype,
        )
        status_feature = torch.cat([command_onehot, ego_status], dim=-1)
        status_encoding = self.status_encoder(status_feature)
        agents_query = self._pad_agent_queries(track_query_embeddings, batch_size, device, dtype)
        agents_query = self.agent_query_proj(agents_query)

        ego_query = self.ego_query_fuser(torch.cat([sdc_embedding, status_encoding], dim=-1))
        ego_query = ego_query[:, None, :]
        return ego_query, agents_query, status_encoding[:, None, :]

    def _build_bev_feature(self, bev_embed, bev_pos):
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1).contiguous()
            bev_feat = bev_embed + bev_pos
        else:
            bev_feat = bev_embed
        if bev_feat.dim() == 3 and bev_feat.shape[1] != self.bev_h * self.bev_w:
            bev_feat = bev_feat.permute(1, 0, 2).contiguous()
        if bev_feat.dim() != 3:
            raise ValueError('bev_embed must have shape (HW, B, C) or (B, HW, C)')
        bev_feat = bev_feat.permute(0, 2, 1).contiguous().view(-1, self.embed_dims, self.bev_h, self.bev_w)
        if self.with_adapter:
            bev_feat = bev_feat + self.bev_adapter(bev_feat)
        return bev_feat

    def _prepare_targets(self, sdc_planning, sdc_planning_mask):
        if isinstance(sdc_planning, (list, tuple)):
            sdc_planning = sdc_planning[0]
        if isinstance(sdc_planning_mask, (list, tuple)):
            sdc_planning_mask = sdc_planning_mask[0]
        if sdc_planning.dim() == 4:
            sdc_planning = sdc_planning[0]
        if sdc_planning.dim() == 2:
            sdc_planning = sdc_planning.unsqueeze(0)
        if sdc_planning_mask.dim() == 4:
            sdc_planning_mask = sdc_planning_mask[0]
        if sdc_planning_mask.dim() == 3:
            sdc_planning_mask = torch.any(sdc_planning_mask[..., :2] > 0, dim=-1)
        sdc_planning = sdc_planning[:, :self.planning_steps]
        sdc_planning_mask = sdc_planning_mask[:, :self.planning_steps].to(dtype=sdc_planning.dtype)
        return sdc_planning, sdc_planning_mask

    def _decode(self, bev_feature, ego_query, agents_query, status_encoding, noisy_traj_points, timesteps):
        bs, ego_fut_mode, _, _ = noisy_traj_points.shape
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=self.embed_dims)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)
        time_embed = self.time_mlp(timesteps).view(bs, 1, -1) + status_encoding
        return self.diff_decoder(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            agents_query,
            ego_query,
            time_embed,
        )

    def forward_train(self,
                      bev_embed,
                      outs_motion={},
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None):
        bev_pos = outs_motion['bev_pos']
        bev_feature = self._build_bev_feature(bev_embed, bev_pos)
        ego_query, agents_query, status_encoding = self._build_condition(outs_motion, command)
        bs = ego_query.shape[0]
        device = ego_query.device

        plan_anchor = self.plan_anchor.to(device=device, dtype=ego_query.dtype).unsqueeze(0).repeat(bs, 1, 1, 1)
        norm_anchor = self._norm_xy(plan_anchor)
        timesteps = torch.randint(0, self.train_noise_timesteps, (bs,), device=device).long()
        noise = torch.randn(norm_anchor.shape, device=device, dtype=norm_anchor.dtype)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=norm_anchor,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = self._denorm_xy(torch.clamp(noisy_traj_points, min=-1.0, max=1.0))

        poses_reg_list, poses_cls_list = self._decode(
            bev_feature,
            ego_query,
            agents_query,
            status_encoding,
            noisy_traj_points,
            timesteps,
        )
        poses_reg = poses_reg_list[-1]
        poses_cls = poses_cls_list[-1]
        sdc_traj_all = self._select_best_mode(poses_reg, poses_cls)
        outs_planning = dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
            sdc_traj_modes=poses_reg,
            sdc_traj_cls=poses_cls,
            plan_anchor=plan_anchor,
            decoder_reg_list=poses_reg_list,
            decoder_cls_list=poses_cls_list,
        )
        losses = self.loss(sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes)
        return dict(losses=losses, outs_motion=outs_planning)

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None, drivable_pred=None):
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']
        bev_feature = self._build_bev_feature(bev_embed, bev_pos)
        ego_query, agents_query, status_encoding = self._build_condition(outs_motion, command)
        bs = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20.0 / float(max(self.inference_steps, 1))
        roll_timesteps = (np.arange(0, self.inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device=device)

        plan_anchor = self.plan_anchor.to(device=device, dtype=ego_query.dtype).unsqueeze(0).repeat(bs, 1, 1, 1)
        img = self._norm_xy(plan_anchor)
        noise = torch.randn(img.shape, device=device, dtype=img.dtype)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        poses_reg, poses_cls = None, None

        for timestep in roll_timesteps:
            noisy_traj_points = self._denorm_xy(torch.clamp(img, min=-1.0, max=1.0))
            timesteps = timestep.expand(bs)
            poses_reg_list, poses_cls_list = self._decode(
                bev_feature,
                ego_query,
                agents_query,
                status_encoding,
                noisy_traj_points,
                timesteps,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            img = self.diffusion_scheduler.step(
                model_output=self._norm_xy(poses_reg),
                timestep=timestep,
                sample=img,
            ).prev_sample

        sdc_traj_all = self._select_best_mode(poses_reg, poses_cls)
        if self.use_col_optim and not self.training:
            assert occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)
            sdc_traj_all = self.drivable_optimization(sdc_traj_all, drivable_pred)

        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
            sdc_traj_modes=poses_reg,
            sdc_traj_cls=poses_cls,
        )

    def _select_best_mode(self, poses_reg, poses_cls):
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[:, None, None, None].repeat(1, 1, self.planning_steps, 2)
        return torch.gather(poses_reg, 1, mode_idx).squeeze(1)

    def _diffusion_loss(self, outs_planning, sdc_planning, sdc_planning_mask):
        target_traj, target_mask = self._prepare_targets(sdc_planning, sdc_planning_mask)
        target_traj = target_traj.to(outs_planning['sdc_traj_modes'].device)
        target_mask = target_mask.to(outs_planning['sdc_traj_modes'].device)
        target_xy = target_traj[..., :2]

        plan_anchor = outs_planning['plan_anchor']
        dist = torch.linalg.norm(target_xy.unsqueeze(1) - plan_anchor, dim=-1)
        dist = (dist * target_mask.unsqueeze(1)).sum(dim=-1) / (target_mask.sum(dim=-1, keepdim=True) + 1e-5)
        cls_target = torch.argmin(dist, dim=-1)

        loss_dict = {}
        for layer_idx, (poses_reg, poses_cls) in enumerate(zip(
                outs_planning['decoder_reg_list'],
                outs_planning['decoder_cls_list'])):
            gather_idx = cls_target[:, None, None, None].repeat(1, 1, self.planning_steps, 2)
            best_reg = torch.gather(poses_reg, 1, gather_idx).squeeze(1)
            target_classes_onehot = torch.zeros_like(poses_cls)
            target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

            loss_cls = self.loss_diffusion_cls_weight * py_sigmoid_focal_loss(
                poses_cls,
                target_classes_onehot,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
            )
            reg_mask = target_mask.unsqueeze(-1).expand_as(target_xy)
            reg_err = torch.abs(best_reg - target_xy) * reg_mask
            loss_reg = self.loss_diffusion_reg_weight * (
                reg_err.sum() / (reg_mask.sum() + 1e-5)
            )
            loss_dict['loss_diffusion_cls_{}'.format(layer_idx)] = loss_cls
            loss_dict['loss_diffusion_reg_{}'.format(layer_idx)] = loss_reg
        return loss_dict, target_xy, target_mask

    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
        diffusion_losses, target_xy, target_mask = self._diffusion_loss(
            outs_planning,
            sdc_planning,
            sdc_planning_mask,
        )
        loss_dict = dict(diffusion_losses)
        if not self.train_aux_planning_losses:
            return loss_dict

        if future_gt_bbox is not None:
            target_traj, _ = self._prepare_targets(sdc_planning, sdc_planning_mask)
            target_traj = target_traj.to(outs_planning['sdc_traj_all'].device)
            for i in range(len(self.loss_collision)):
                loss_collision = self.loss_collision[i](
                    outs_planning['sdc_traj_all'],
                    target_traj[..., :3],
                    target_mask,
                    future_gt_bbox[0][1:self.planning_steps + 1],
                )
                loss_dict['loss_collision_{}'.format(i)] = loss_collision
        loss_ade = self.loss_planning(outs_planning['sdc_traj_all'], target_xy, target_mask)
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict
