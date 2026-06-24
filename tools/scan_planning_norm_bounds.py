#!/usr/bin/env python
"""Scan SPD planning GT distribution and suggest diffusion xy normalization bounds."""
import argparse
import importlib
import json
import os
import pickle
import sys

import numpy as np
from mmcv import Config
from mmdet3d.core.bbox import Box3DMode
from nuscenes import NuScenes

importlib.import_module("projects.mmdet3d_plugin")
from projects.mmdet3d_plugin.datasets.data_utils.spd_trajectory_api import SPDTraj


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="projects/configs_e2e_univ2x/univ2x_coop_e2e.py",
        help="UniV2X config (for paths and planning_steps).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Which ann pkl split to scan.",
    )
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=None,
        help="Override planning horizon steps (default: from config).",
    )
    parser.add_argument(
        "--min-valid-steps",
        type=int,
        default=None,
        help="Require at least this many valid future steps (default: planning_steps).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Limit number of frames to scan (-1 = all).",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower percentile for suggested bounds.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper percentile for suggested bounds.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.10,
        help="Extra margin as fraction of (p_high - p_low).",
    )
    parser.add_argument(
        "--margin-min",
        type=float,
        default=2.0,
        help="Minimum absolute margin (meters) per axis.",
    )
    parser.add_argument(
        "--anchor-path",
        default="data/plan_anchors/spd_plan_anchors_20x10.npy",
        help="Optional anchor npy for comparison.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional path to save stats json.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="NuScenes version under data_root.",
    )
    return parser.parse_args()


def load_infos(ann_file):
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    return data


def build_traj_api(nusc, cfg, planning_steps):
    return SPDTraj(
        nusc,
        predict_steps=cfg.predict_steps,
        planning_steps=planning_steps,
        past_steps=cfg.past_steps,
        fut_steps=cfg.fut_steps,
        with_velocity=True,
        CLASSES=cfg.class_names,
        box_mode_3d=Box3DMode.LIDAR,
        use_nonlinear_optimizer=cfg.use_nonlinear_optimizer,
    )


def collect_planning_points(traj_api, infos, planning_steps, min_valid_steps, max_samples):
    xy_list = []
    yaw_list = []
    valid_traj_count = 0
    skipped_short = 0
    command_hist = {}

    n_total = len(infos) if max_samples < 0 else min(max_samples, len(infos))
    for idx in range(n_total):
        info = infos[idx]
        planning, mask, command = traj_api.get_sdc_planning_label(info["token"])
        planning = np.squeeze(planning).astype(np.float32)
        mask = np.squeeze(mask).astype(np.float32)

        if planning.ndim == 3:
            planning = planning[0]
        if mask.ndim == 3:
            mask = mask[0]

        valid = mask[..., :2].any(axis=-1)[:planning_steps]
        n_valid = int(valid.sum())
        if n_valid < min_valid_steps:
            skipped_short += 1
            continue

        valid_traj_count += 1
        command_hist[int(command)] = command_hist.get(int(command), 0) + 1

        traj_xy = planning[:planning_steps, :2][valid]
        traj_yaw = planning[:planning_steps, 2][valid]
        xy_list.append(traj_xy)
        yaw_list.append(traj_yaw)

        if (idx + 1) % 200 == 0:
            print(f"  scanned {idx + 1}/{n_total}, valid_trajs={valid_traj_count}", flush=True)

    if not xy_list:
        raise RuntimeError("No valid planning trajectories collected.")

    xy = np.concatenate(xy_list, axis=0)
    yaw = np.concatenate(yaw_list, axis=0)
    meta = dict(
        scanned_frames=n_total,
        valid_trajs=valid_traj_count,
        skipped_short=skipped_short,
        valid_points=int(xy.shape[0]),
        command_hist=command_hist,
    )
    return xy, yaw, meta


def summarize_1d(values, percentiles):
    stats = {
        "count": int(values.shape[0]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    for p in percentiles:
        stats[f"p{p:g}"] = float(np.percentile(values, p))
    return stats


def suggest_bounds_from_xy(xy, p_low, p_high, margin_ratio, margin_min):
    bounds = []
    for dim, axis_name in enumerate(["x", "y"]):
        vals = xy[:, dim]
        lo = float(np.percentile(vals, p_low))
        hi = float(np.percentile(vals, p_high))
        span = max(hi - lo, 1e-3)
        margin = max(span * margin_ratio, margin_min)
        bounds.append((lo - margin, hi + margin))
    return bounds


def load_anchor_stats(anchor_path):
    if not anchor_path or not os.path.exists(anchor_path):
        return None
    anchors = np.load(anchor_path).astype(np.float32)
    return {
        "path": anchor_path,
        "shape": list(anchors.shape),
        "min": anchors.min(axis=(0, 1)).tolist(),
        "max": anchors.max(axis=(0, 1)).tolist(),
    }


def print_stats_table(xy_stats, yaw_stats, anchor_stats, suggested, dd_bounds):
    print("\n=== SPD planning GT statistics (ego/lidar frame) ===")
    print(f"valid xy points: {xy_stats['count']}")
    print(f"  x  min/max: {xy_stats['x_min']:.4f} / {xy_stats['x_max']:.4f}")
    print(f"  y  min/max: {xy_stats['y_min']:.4f} / {xy_stats['y_max']:.4f}")
    print(f"  x  mean/std: {xy_stats['x_mean']:.4f} / {xy_stats['x_std']:.4f}")
    print(f"  y  mean/std: {xy_stats['y_mean']:.4f} / {xy_stats['y_std']:.4f}")
    print(f"  x  p1/p50/p99: {xy_stats['x_p1']:.4f} / {xy_stats['x_p50']:.4f} / {xy_stats['x_p99']:.4f}")
    print(f"  y  p1/p50/p99: {xy_stats['y_p1']:.4f} / {xy_stats['y_p50']:.4f} / {xy_stats['y_p99']:.4f}")

    print("\n=== yaw (rad) ===")
    print(f"  min/max: {yaw_stats['min']:.4f} / {yaw_stats['max']:.4f}")
    print(f"  p1/p99:  {yaw_stats['p1']:.4f} / {yaw_stats['p99']:.4f}")

    if anchor_stats:
        print("\n=== plan anchors (for comparison) ===")
        print(f"  path: {anchor_stats['path']}")
        print(f"  shape: {anchor_stats['shape']}")
        print(f"  xy min: {anchor_stats['min']}")
        print(f"  xy max: {anchor_stats['max']}")

    print("\n=== DiffusionDrive NAVSIM hardcoded (reference only) ===")
    print(f"  x: [{dd_bounds['x'][0]}, {dd_bounds['x'][1]}]")
    print(f"  y: [{dd_bounds['y'][0]}, {dd_bounds['y'][1]}]")
    print(f"  heading: [{dd_bounds['heading'][0]}, {dd_bounds['heading'][1]}]")

    (x_min, x_max), (y_min, y_max) = suggested
    print("\n=== suggested xy_norm_bounds (GT percentile + margin) ===")
    print(f"  xy_norm_bounds=(({x_min:.4f}, {y_min:.4f}), ({x_max:.4f}, {y_max:.4f}))")
    print("\n  # paste into planning_head config:")
    print(f"  xy_norm_bounds=[[{x_min:.4f}, {y_min:.4f}], [{x_max:.4f}, {y_max:.4f}]],")


def summarize_xy_per_axis(xy, percentiles):
    stats = {"count": int(xy.shape[0])}
    for dim, name in enumerate(["x", "y"]):
        vals = xy[:, dim]
        stats[f"{name}_min"] = float(np.min(vals))
        stats[f"{name}_max"] = float(np.max(vals))
        stats[f"{name}_mean"] = float(np.mean(vals))
        stats[f"{name}_std"] = float(np.std(vals))
        for p in percentiles:
            stats[f"{name}_p{p:g}"] = float(np.percentile(vals, p))
    return stats


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    info_root = cfg.info_root
    ann_file = (
        os.path.join(info_root, "spd_infos_temporal_train.pkl")
        if args.split == "train"
        else os.path.join(info_root, "spd_infos_temporal_val.pkl")
    )
    data_root = cfg.data_root
    planning_steps = args.planning_steps or cfg.planning_steps
    min_valid_steps = args.min_valid_steps or planning_steps
    percentiles = [0.5, 1, 5, 50, 95, 99, 99.5]

    print(f"[1/4] load infos: {ann_file}", flush=True)
    infos = load_infos(ann_file)
    print(f"  frames in pkl: {len(infos)}", flush=True)

    print(f"[2/4] init NuScenes ({args.version}) @ {data_root}", flush=True)
    nusc = NuScenes(version=args.version, dataroot=data_root, verbose=False)

    print("[3/4] init SPDTraj and scan planning GT ...", flush=True)
    traj_api = build_traj_api(nusc, cfg, planning_steps)
    xy, yaw, meta = collect_planning_points(
        traj_api, infos, planning_steps, min_valid_steps, args.max_samples
    )

    print("[4/4] summarize ...", flush=True)
    xy_stats = summarize_xy_per_axis(xy, percentiles)
    yaw_stats = summarize_1d(yaw, percentiles)
    suggested = suggest_bounds_from_xy(
        xy, args.percentile_low, args.percentile_high, args.margin_ratio, args.margin_min
    )
    anchor_stats = load_anchor_stats(args.anchor_path)
    dd_bounds = {
        "x": (-1.2, 55.7),
        "y": (-20.0, 26.0),
        "heading": (-2.0, 1.9),
    }

    print(f"\nmeta: {meta}", flush=True)
    print_stats_table(xy_stats, yaw_stats, anchor_stats, suggested, dd_bounds)

    if anchor_stats:
        anchors = np.load(args.anchor_path).astype(np.float32)
        anchor_bounds = suggest_bounds_from_xy(
            anchors.reshape(-1, 2), 0, 100, args.margin_ratio, args.margin_min
        )
        print("\n=== suggested bounds from anchors only (min/max + margin) ===")
        print(f"  {anchor_bounds}")

    result = dict(
        meta=meta,
        planning_steps=planning_steps,
        xy_stats=xy_stats,
        yaw_stats=yaw_stats,
        suggested_xy_norm_bounds={
            "percentile_low": args.percentile_low,
            "percentile_high": args.percentile_high,
            "margin_ratio": args.margin_ratio,
            "margin_min": args.margin_min,
            "xy_min": [suggested[0][0], suggested[1][0]],
            "xy_max": [suggested[0][1], suggested[1][1]],
        },
        anchor_stats=anchor_stats,
        diffusiondrive_reference_bounds=dd_bounds,
    )
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nsaved: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
