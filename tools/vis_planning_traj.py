#!/usr/bin/env python
"""Visualize planning trajectories (GT vs prediction) in BEV."""
import argparse
import importlib
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

importlib.import_module("projects.mmdet3d_plugin")
logging.getLogger("mmcv").setLevel(logging.ERROR)

COMMAND_NAMES = {0: "RIGHT", 1: "LEFT", 2: "FORWARD"}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize UniV2X planning trajectories")
    parser.add_argument("--config", default="projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    parser.add_argument("--ckpt", default="ckpts/univ2x_coop_e2e_stg2.pth")
    parser.add_argument(
        "--planning-ckpt",
        default="work_dirs/plan_overfit/planning_head_overfit.pth",
        help="overfit 后的 planning_head 权重；留空则只用 base ckpt",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--out-dir", default="work_dirs/plan_overfit/vis")
    parser.add_argument("--queue-length", type=int, default=1)
    parser.add_argument("--planning-steps", type=int, default=10)
    parser.add_argument("--show-modes", action="store_true", help="overlay 20 diffusion modes")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-5, 55])
    parser.add_argument("--ylim", type=float, nargs=2, default=[-25, 25])
    return parser.parse_args()


def parse_indices(indices_str, dataset_len):
    indices = []
    for part in indices_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    indices = sorted(set(i for i in indices if 0 <= i < dataset_len))
    if not indices:
        raise ValueError(f"no valid indices in '{indices_str}' for dataset length {dataset_len}")
    return indices


def reset_scene_state(model):
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


def unwrap_data(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return value
        return unwrap_data(value[0])
    return value


def to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_xy(traj, steps):
    arr = to_numpy(traj)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] >= 2:
        arr = arr[..., :2]
    arr = arr.reshape(-1, arr.shape[-1])[:steps]
    return arr


def extract_mask(mask, steps):
    arr = to_numpy(mask)
    if arr is None:
        return np.ones(steps, dtype=bool)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        valid = arr.any(axis=-1)
    else:
        valid = arr.astype(bool)
    valid = valid.reshape(-1)[:steps]
    if valid.shape[0] < steps:
        pad = np.zeros(steps - valid.shape[0], dtype=bool)
        valid = np.concatenate([valid, pad], axis=0)
    return valid


def compute_ade(pred_xy, gt_xy, valid):
    if pred_xy is None or gt_xy is None or valid.sum() == 0:
        return float("nan")
    n = min(len(pred_xy), len(gt_xy), len(valid))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]
    valid = valid[:n]
    err = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    return float((err * valid).sum() / (valid.sum() + 1e-5))


def plot_trajectory(
    gt_xy,
    pred_xy,
    valid,
    command,
    ade,
    out_path,
    sample_idx,
    show_modes=False,
    modes_xy=None,
    xlim=(-5, 55),
    ylim=(-25, 25),
):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax.axvline(0.0, color="#cccccc", linewidth=0.8)
    ax.scatter([0.0], [0.0], c="black", s=40, zorder=5, label="ego")

    if show_modes and modes_xy is not None:
        for mode_idx, mode_traj in enumerate(modes_xy):
            label = "modes" if mode_idx == 0 else None
            ax.plot(
                mode_traj[:, 0],
                mode_traj[:, 1],
                color="#9ecae1",
                alpha=0.25,
                linewidth=1.0,
                label=label,
            )

    if gt_xy is not None and valid.sum() > 0:
        gt_valid = gt_xy[valid]
        ax.plot(
            gt_valid[:, 0],
            gt_valid[:, 1],
            color="#2ca02c",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="GT",
        )

    if pred_xy is not None:
        pred_plot = pred_xy[valid] if valid.sum() > 0 else pred_xy
        ax.plot(
            pred_plot[:, 0],
            pred_plot[:, 1],
            color="#d62728",
            linewidth=2.5,
            marker="s",
            markersize=4,
            label="pred",
        )

    cmd_name = COMMAND_NAMES.get(int(command), str(command))
    ax.set_title(
        f"sample={sample_idx} | command={cmd_name} | ADE={ade:.2f}m",
        fontsize=12,
    )
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, lateral)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_model_from_cfg(cfg, ckpt, planning_ckpt=None):
    model = build_model(
        cfg.model_ego_agent,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()
    load_checkpoint(model, ckpt, map_location="cpu", strict=False)
    if planning_ckpt:
        payload = torch.load(planning_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys when loading planning head: {unexpected}")
        print(f"loaded planning head from {planning_ckpt}", flush=True)
        if payload.get("meta"):
            print(f"planning ckpt meta: {payload['meta']}", flush=True)
    model.cuda()
    model.eval()
    return model


def run_inference(model, dataset, sample_idx, planning_steps, show_modes):
    data = collate([dataset[sample_idx]], samples_per_gpu=1)
    data = scatter(data, [0])[0]
    ego_data = data["ego_agent_data"]
    captured = {}

    original_forward_train = model.planning_head.forward_train

    def forward_train_wrapper(*args, **kwargs):
        ret = original_forward_train(*args, **kwargs)
        captured["planning"] = ret["outs_motion"]
        return ret

    reset_scene_state(model)
    model.planning_head.forward_train = forward_train_wrapper
    try:
        with torch.no_grad():
            model(
                return_loss=True,
                other_agent_results=None,
                img_metas=ego_data["img_metas"],
                **{k: v for k, v in ego_data.items() if k != "img_metas"},
            )
    finally:
        model.planning_head.forward_train = original_forward_train

    planning = captured["planning"]
    pred_xy = extract_xy(planning.get("sdc_traj_all", planning.get("sdc_traj")), planning_steps)
    gt_xy = extract_xy(ego_data["sdc_planning"], planning_steps)
    valid = extract_mask(ego_data["sdc_planning_mask"], planning_steps)

    modes_xy = None
    if show_modes and "sdc_traj_modes" in planning:
        modes = to_numpy(planning["sdc_traj_modes"])
        modes = np.squeeze(modes)
        if modes.ndim == 4:
            modes = modes[0]
        modes_xy = [modes[i, :, :2] for i in range(modes.shape[0])]

    command = int(np.squeeze(to_numpy(unwrap_data(ego_data["command"]))))
    img_metas = unwrap_data(ego_data["img_metas"])
    if isinstance(img_metas, dict):
        token = img_metas[max(img_metas.keys())].get("sample_idx", sample_idx)
    elif isinstance(img_metas, list):
        token = img_metas[-1].get("sample_idx", sample_idx)
    else:
        token = sample_idx
    ade = compute_ade(pred_xy, gt_xy, valid)
    return {
        "sample_idx": sample_idx,
        "token": str(token),
        "command": command,
        "ade": ade,
        "pred_xy": pred_xy,
        "gt_xy": gt_xy,
        "valid": valid,
        "modes_xy": modes_xy,
    }


def make_summary_grid(records, out_path, xlim, ylim):
    n = len(records)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]

    for ax in axes.ravel():
        ax.axis("off")

    for i, rec in enumerate(records):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("on")
        valid = rec["valid"]
        gt_xy, pred_xy = rec["gt_xy"], rec["pred_xy"]
        if gt_xy is not None and valid.sum() > 0:
            gt_valid = gt_xy[valid]
            ax.plot(gt_valid[:, 0], gt_valid[:, 1], "g-o", linewidth=1.5, markersize=2)
        if pred_xy is not None:
            pred_plot = pred_xy[valid] if valid.sum() > 0 else pred_xy
            ax.plot(pred_plot[:, 0], pred_plot[:, 1], "r-s", linewidth=1.5, markersize=2)
        ax.scatter([0], [0], c="k", s=8)
        cmd = COMMAND_NAMES.get(rec["command"], str(rec["command"]))
        ax.set_title(f"#{rec['sample_idx']} {cmd} ADE={rec['ade']:.2f}m", fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "需要 GPU"

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    split_cfg = getattr(cfg.data, args.split)
    split_cfg.queue_length = args.queue_length
    dataset = build_dataset(split_cfg)

    indices = parse_indices(args.indices, len(dataset))
    planning_ckpt = args.planning_ckpt if args.planning_ckpt else None
    if planning_ckpt and not os.path.isfile(planning_ckpt):
        raise FileNotFoundError(
            f"planning ckpt not found: {planning_ckpt}\n"
            "请先重跑 overfit（会自动保存 planning_head_overfit.pth），"
            "或用 --planning-ckpt 指定路径。"
        )

    print(f"[1/3] build model, visualize {len(indices)} samples from {args.split}", flush=True)
    model = build_model_from_cfg(cfg, args.ckpt, planning_ckpt)

    records = []
    print("[2/3] inference + plot ...", flush=True)
    for sample_idx in indices:
        rec = run_inference(
            model, dataset, sample_idx, args.planning_steps, args.show_modes
        )
        out_path = os.path.join(args.out_dir, f"sample_{sample_idx:04d}.png")
        plot_trajectory(
            gt_xy=rec["gt_xy"],
            pred_xy=rec["pred_xy"],
            valid=rec["valid"],
            command=rec["command"],
            ade=rec["ade"],
            out_path=out_path,
            sample_idx=sample_idx,
            show_modes=args.show_modes,
            modes_xy=rec["modes_xy"],
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
        )
        records.append(rec)
        print(
            f"  sample {sample_idx:4d} | command={COMMAND_NAMES.get(rec['command'], rec['command'])} "
            f"| ADE={rec['ade']:.3f}m -> {out_path}",
            flush=True,
        )

    summary_path = os.path.join(args.out_dir, "summary_grid.png")
    make_summary_grid(records, summary_path, tuple(args.xlim), tuple(args.ylim))

    metrics = [
        {
            "sample_idx": rec["sample_idx"],
            "token": rec["token"],
            "command": COMMAND_NAMES.get(rec["command"], str(rec["command"])),
            "ade": rec["ade"],
        }
        for rec in records
    ]
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "planning_ckpt": planning_ckpt,
                "split": args.split,
                "mean_ade": float(np.nanmean([m["ade"] for m in metrics])),
                "samples": metrics,
            },
            f,
            indent=2,
        )

    print("[3/3] done", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
