_base_ = ["./univ2x_coop_e2e.py"]

# Full train-split fine-tuning for the migrated diffusion planner.
# Upstream perception/map/motion/occ modules stay frozen and deterministic;
# only the ego-agent planning head is optimized.
work_dir = "projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain"
load_from = None
resume_from = (
    "projects/work_dirs_e2e_univ2x/"
    "univ2x_coop_e2e_diffusion_plan_fulltrain/epoch_3.pth"
)

trainable_param_prefixes = ("model_ego_agent.planning_head.",)
custom_hooks = [
    dict(
        type="TrainableParamPrefixHook",
        trainable_param_prefixes=trainable_param_prefixes,
        priority="VERY_HIGH",
    ),
    dict(
        type="BestLatestPlanningCheckpointHook",
        interval=1,
        metric_keys=(
            "planning.loss_diffusion_reg_0",
            "planning.loss_diffusion_reg_1",
        ),
        priority="NORMAL",
    ),
]

optimizer = dict(type="AdamW", lr=5e-5, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 20
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
# Only keep latest.pth and best.pth via BestLatestPlanningCheckpointHook.
checkpoint_config = None

# 4 卡 x 8 workers 容易把 CPU 内存打满，导致 worker 被 OOM killer 杀掉
data = dict(workers_per_gpu=2)

# Match DiffusionDrive's planner training objective: only optimize the
# anchor-matched diffusion classification/regression losses.
model_ego_agent = dict(
    planning_head=dict(
        xy_norm_bounds=[[-4.0328, -17.6693], [54.2865, 20.0801]],
        train_aux_planning_losses=False,
    ),
)
