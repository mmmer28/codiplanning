#!/usr/bin/env python
"""Generate planning trajectory anchors from SPD training labels."""
import argparse
import importlib
import os

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

importlib.import_module("projects.mmdet3d_plugin")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    parser.add_argument("--out", default="data/plan_anchors/spd_plan_anchors_20x10.npy")
    parser.add_argument("--num-modes", type=int, default=20)
    parser.add_argument("--planning-steps", type=int, default=10)
    parser.add_argument("--min-valid-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def collect_planning_trajs(dataset, planning_steps, min_valid_steps, max_samples):
    trajs = []
    commands = []
    n = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))
    for idx in range(n):
        ann = dataset.get_ann_info(idx)
        planning = np.asarray(ann["sdc_planning"], dtype=np.float32)
        mask = np.asarray(ann["sdc_planning_mask"], dtype=np.float32)

        planning = np.squeeze(planning)[..., :planning_steps, :2]
        mask = np.squeeze(mask)[..., :planning_steps, :]
        if planning.ndim == 3:
            planning = planning[0]
        if mask.ndim == 3:
            mask = mask[0]

        valid = mask[..., :2].any(axis=-1)
        if int(valid.sum()) < min_valid_steps:
            continue
        if not valid[:planning_steps].all():
            continue
        trajs.append(planning[:planning_steps])
        commands.append(int(ann.get("command", -1)))

        if (idx + 1) % 1000 == 0:
            print(f"processed {idx + 1}/{n}, collected {len(trajs)}", flush=True)

    if not trajs:
        raise RuntimeError("No valid planning trajectories collected.")
    return np.stack(trajs, axis=0).astype(np.float32), np.asarray(commands, dtype=np.int64)


def init_kmeans_pp(x, k, rng):
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.randint(x.shape[0])]
    dist_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(dist_sq.sum())
        if total <= 1e-12:
            centers[i] = x[rng.randint(x.shape[0])]
            continue
        probs = dist_sq / total
        centers[i] = x[rng.choice(x.shape[0], p=probs)]
        dist_sq = np.minimum(dist_sq, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def run_kmeans(trajs, num_modes, iters, seed):
    x = trajs.reshape(trajs.shape[0], -1)
    rng = np.random.RandomState(seed)
    centers = init_kmeans_pp(x, num_modes, rng)

    for it in range(iters):
        dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for mode in range(num_modes):
            assigned = x[labels == mode]
            if len(assigned) > 0:
                new_centers[mode] = assigned.mean(axis=0)
            else:
                farthest = np.argmax(dist.min(axis=1))
                new_centers[mode] = x[farthest]
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if (it + 1) % 10 == 0 or it == 0:
            mean_dist = float(np.sqrt(dist.min(axis=1)).mean())
            print(f"kmeans iter {it + 1:03d}: shift={shift:.4f}, mean_l2={mean_dist:.4f}", flush=True)
        if shift < 1e-4:
            break

    anchors = centers.reshape(num_modes, trajs.shape[1], 2)
    order = np.argsort(anchors[:, -1, 0])
    return anchors[order].astype(np.float32)


def summarize(trajs, anchors, commands):
    x = trajs.reshape(trajs.shape[0], -1)
    c = anchors.reshape(anchors.shape[0], -1)
    dist = np.sqrt(np.sum((x[:, None, :] - c[None, :, :]) ** 2, axis=-1))
    nearest = np.argmin(dist, axis=1)
    print("\n=== anchor summary ===", flush=True)
    print(f"num_trajs: {len(trajs)}", flush=True)
    print(f"nearest flattened L2 mean: {dist.min(axis=1).mean():.4f}", flush=True)
    print(f"nearest flattened L2 p90:  {np.percentile(dist.min(axis=1), 90):.4f}", flush=True)
    print(f"anchor final x range: {anchors[:, -1, 0].min():.2f}..{anchors[:, -1, 0].max():.2f}", flush=True)
    print(f"anchor final y range: {anchors[:, -1, 1].min():.2f}..{anchors[:, -1, 1].max():.2f}", flush=True)
    for cmd in sorted(set(commands.tolist())):
        print(f"command {cmd}: {(commands == cmd).sum()} trajs", flush=True)
    counts = np.bincount(nearest, minlength=anchors.shape[0])
    print(f"cluster counts: {counts.tolist()}", flush=True)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    train_cfg = cfg.data.train.copy()
    train_cfg.queue_length = 1

    print("[1/3] build train dataset ...", flush=True)
    dataset = build_dataset(train_cfg)

    print("[2/3] collect planning trajectories ...", flush=True)
    trajs, commands = collect_planning_trajs(
        dataset,
        args.planning_steps,
        args.min_valid_steps,
        args.max_samples,
    )

    print("[3/3] run kmeans ...", flush=True)
    anchors = run_kmeans(trajs, args.num_modes, args.iters, args.seed)
    summarize(trajs, anchors, commands)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, anchors)
    print(f"\nsaved anchors: {args.out}", flush=True)
    print(f"shape: {anchors.shape}", flush=True)


if __name__ == "__main__":
    main()
