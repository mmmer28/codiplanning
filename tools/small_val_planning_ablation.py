#!/usr/bin/env python
"""Small val ablation: command-only (zero ego status) vs +status."""
import argparse
import importlib
import json
import logging
import os
import time

import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

importlib.import_module("projects.mmdet3d_plugin")
from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head_plugin import PlanningMetric

logging.getLogger("mmcv").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    parser.add_argument("--ckpt", default="ckpts/univ2x_coop_e2e_stg2.pth")
    parser.add_argument(
        "--planning-ckpt",
        default="work_dirs/plan_overfit/planning_head_overfit.pth",
        help="overfit planning head;留空则只用 base ckpt",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--queue-length", type=int, default=5)
    parser.add_argument(
        "--print-interval",
        type=int,
        default=1,
        help="print progress every N frames; default 1 for easier hang diagnosis",
    )
    parser.add_argument(
        "--train-style-val",
        action="store_true",
        help=(
            "deprecated: 当前 val/test 使用 test_pipeline，不能直接走 train-style queue；"
            "保留参数仅用于给出明确报错。"
        ),
    )
    parser.add_argument("--out", default="work_dirs/plan_overfit/small_val_ablation.json")
    return parser.parse_args()


def load_model(cfg, ckpt, planning_ckpt=None):
    model = build_model(
        cfg.model_ego_agent,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()
    load_checkpoint(model, ckpt, map_location="cpu", strict=False)
    if planning_ckpt:
        payload = torch.load(planning_ckpt, map_location="cpu")
        model.load_state_dict(payload["state_dict"], strict=False)
        print(f"loaded planning head: {planning_ckpt}", flush=True)
    model.cuda()
    model.eval()
    return model


def install_zero_status(model):
    def _zero(device, dtype):
        return torch.zeros(1, 4, device=device, dtype=dtype)

    def zero_train(l2g_t, l2g_r_mat, timestamp, device, dtype):
        return _zero(device, dtype)

    def zero_test(l2g_t, l2g_r_mat, timestamp, is_new_scene, device, dtype):
        return _zero(device, dtype)

    orig = (model._build_train_ego_status, model._build_test_ego_status)
    model._build_train_ego_status = zero_train
    model._build_test_ego_status = zero_test
    return orig


def restore_status_builders(model, orig):
    model._build_train_ego_status, model._build_test_ego_status = orig


def reset_runtime_state(model):
    model.prev_frame_info["prev_bev"] = None
    model.prev_frame_info["scene_token"] = None
    model.prev_frame_info["prev_pos"] = 0
    model.prev_frame_info["prev_angle"] = 0
    model.prev_frame_info["planning_prev_l2g_t"] = None
    model.prev_frame_info["planning_prev_timestamp"] = None
    model.prev_frame_info["planning_prev_velocity"] = None
    model.test_track_instances = None
    model.scene_token = None
    model.prev_bev = None


def run_pass(model, dataset, indices, planning_steps, zero_status, phase_name, print_interval):
    if zero_status:
        orig = install_zero_status(model)
    else:
        orig = None

    reset_runtime_state(model)
    metric = PlanningMetric(n_future=planning_steps).cuda()

    try:
        for step, idx in enumerate(indices):
            should_print = (
                print_interval > 0
                and (step == 0 or (step + 1) % print_interval == 0 or step + 1 == len(indices))
            )
            frame_start = time.perf_counter()
            if should_print:
                print(
                    f"  [{phase_name}] start frame {step + 1}/{len(indices)} "
                    f"(dataset idx={idx})",
                    flush=True,
                )

            batch = collate([dataset[idx]], samples_per_gpu=1)
            batch = scatter(batch, [0])[0]
            ego = batch["ego_agent_data"]

            with torch.no_grad():
                result = model(
                    return_loss=False,
                    w_label=True,
                    rescale=True,
                    other_agent_results={},
                    img_metas=ego["img_metas"],
                    **{k: v for k, v in ego.items() if k != "img_metas"},
                )

            if "planning" not in result[0]:
                if should_print:
                    elapsed = time.perf_counter() - frame_start
                    print(
                        f"  [{phase_name}] skip frame {step + 1}/{len(indices)} "
                        f"(no planning, {elapsed:.1f}s)",
                        flush=True,
                    )
                continue

            planning = result[0]["planning"]
            pred = planning["result_planning"]["sdc_traj"]
            gt = planning["planning_gt"]["sdc_planning"]
            gt_mask = planning["planning_gt"]["sdc_planning_mask"]
            seg = planning["planning_gt"]["segmentation"]
            drivable = planning["planning_gt"]["drivable_gt"]

            metric(
                pred[:, :planning_steps, :2],
                gt[0][0, :, :planning_steps, :2],
                gt_mask[0][0, :, :planning_steps, :2],
                seg[0][:, 1 : planning_steps + 1],
                drivable,
            )

            if should_print:
                elapsed = time.perf_counter() - frame_start
                print(
                    f"  [{phase_name}] done frame {step + 1}/{len(indices)} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        if orig is not None:
            restore_status_builders(model, orig)

    return metric.compute()


def metrics_to_dict(metrics):
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            out[key] = [float(x) for x in value.detach().cpu().tolist()]
        else:
            out[key] = float(value)
    return out


def summarize_l2(l2_list):
    if not l2_list:
        return {}
    return {
        "L2@1s": l2_list[1] if len(l2_list) > 1 else l2_list[0],
        "L2@3s": l2_list[5] if len(l2_list) > 5 else l2_list[-1],
        "L2@5s": l2_list[9] if len(l2_list) > 9 else l2_list[-1],
        "L2_mean": sum(l2_list) / len(l2_list),
    }


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "需要 GPU"
    if args.train_style_val:
        raise ValueError(
            "--train-style-val is not supported for this ablation script. "
            "The val/test config uses test_pipeline and does not provide "
            "training keys such as gt_labels_3d. Please run without "
            "--train-style-val; test_mode forward_test already computes "
            "ego status from sequential frames."
        )

    cfg = Config.fromfile(args.config)
    split_cfg = cfg.data[args.split].copy()
    split_cfg.pop("samples_per_gpu", None)
    split_cfg.test_mode = True

    dataset = build_dataset(split_cfg)
    planning_steps = getattr(dataset, "planning_steps", 10)
    num_frames = min(args.max_frames, len(dataset))
    indices = list(range(num_frames))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[1/3] load model ({args.ckpt})", flush=True)
    model = load_model(cfg, args.ckpt, args.planning_ckpt or None)

    print(f"[2/3] eval command-only (zero status), {num_frames} frames", flush=True)
    cmd_only = run_pass(
        model,
        dataset,
        indices,
        planning_steps,
        zero_status=True,
        phase_name="command-only",
        print_interval=args.print_interval,
    )

    print(f"[3/3] eval +status, {num_frames} frames", flush=True)
    with_status = run_pass(
        model,
        dataset,
        indices,
        planning_steps,
        zero_status=False,
        phase_name="+status",
        print_interval=args.print_interval,
    )

    cmd_summary = summarize_l2(metrics_to_dict(cmd_only).get("L2", []))
    status_summary = summarize_l2(metrics_to_dict(with_status).get("L2", []))

    result = {
        "split": args.split,
        "num_frames": num_frames,
        "planning_steps": planning_steps,
        "base_ckpt": args.ckpt,
        "planning_ckpt": args.planning_ckpt or None,
        "test_mode": not args.train_style_val,
        "queue_length": args.queue_length if args.train_style_val else None,
        "print_interval": args.print_interval,
        "command_only": metrics_to_dict(cmd_only),
        "with_status": metrics_to_dict(with_status),
        "summary": {
            "command_only": cmd_summary,
            "with_status": status_summary,
            "delta_with_status_minus_command_only": {
                k: status_summary.get(k, 0.0) - cmd_summary.get(k, 0.0)
                for k in cmd_summary.keys()
            },
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== summary (L2, meters) ===", flush=True)
    print(f"command-only: {cmd_summary}", flush=True)
    print(f"+status:      {status_summary}", flush=True)
    print(f"delta:        {result['summary']['delta_with_status_minus_command_only']}", flush=True)
    print(f"saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
