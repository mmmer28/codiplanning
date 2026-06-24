#!/usr/bin/env python
"""Plot planning losses from MMDet .log.json or overfit loss.csv."""
import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt


DEFAULT_KEYS = [
    "planning.loss_ade",
    "planning.loss_diffusion_reg_0",
    "planning.loss_diffusion_cls_0",
    "planning.loss_collision_0",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        nargs="*",
        default=None,
        help="one or more .log.json / loss.csv paths; "
        "default: merge all *.log.json under --work-dir",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/diffusion_plan_fulltrain",
        help="search logs here when --log is omitted",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_KEYS,
        help="metric keys to plot",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output png path; default: <work_dir>/planning_loss.png",
    )
    parser.add_argument(
        "--iters-per-epoch",
        type=int,
        default=0,
        help="global x-axis stride; 0 = infer from epoch-1 max iter",
    )
    parser.add_argument(
        "--split-reg-cls",
        action="store_true",
        help="plot reg and cls metrics in separate figures (different scales)",
    )
    return parser.parse_args()


def find_all_logs(work_dir):
    patterns = [
        os.path.join(work_dir, "*.log.json"),
        os.path.join(work_dir, "loss.csv"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no .log.json or loss.csv found under {work_dir}")
    # Older files first so later runs win on duplicate (epoch, iter).
    return sorted(candidates, key=os.path.getmtime)


def is_training_row(row):
    if "iter" not in row:
        return False
    if row.get("mode") == "train":
        return True
    return "planning.loss_ade" in row


def load_rows(log_path):
    if log_path.endswith(".csv"):
        with open(log_path, newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        rename = {
            "loss_ade": "planning.loss_ade",
            "loss_diffusion_reg_0": "planning.loss_diffusion_reg_0",
            "loss_diffusion_cls_0": "planning.loss_diffusion_cls_0",
            "plan_total": "loss",
        }
        for row in rows:
            for old_key, new_key in rename.items():
                if old_key in row and new_key not in row:
                    row[new_key] = row[old_key]
            if "iter" not in row and "epoch" in row:
                row["iter"] = row["epoch"]
        return rows

    rows = []
    with open(log_path) as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not is_training_row(row):
                continue
            rows.append(row)
    return rows


def merge_rows(row_lists):
    merged = {}
    for rows in row_lists:
        for row in rows:
            epoch = int(row.get("epoch", 1))
            step = int(row["iter"])
            merged[(epoch, step)] = row
    keys = sorted(merged.keys())
    return [merged[key] for key in keys]


def infer_iters_per_epoch(rows):
    epoch1_iters = [int(row["iter"]) for row in rows if int(row.get("epoch", 1)) == 1]
    if epoch1_iters:
        return max(epoch1_iters)
    all_iters = [int(row["iter"]) for row in rows]
    return max(all_iters) if all_iters else 381


def global_step(row, iters_per_epoch):
    epoch = int(row.get("epoch", 1))
    step = int(row["iter"])
    return (epoch - 1) * iters_per_epoch + step


def split_reg_cls_keys(keys):
    reg_keys = [k for k in keys if "reg" in k]
    cls_keys = [k for k in keys if "cls" in k]
    other_keys = [k for k in keys if k not in reg_keys and k not in cls_keys]
    return reg_keys, cls_keys, other_keys


def resolve_out_path(args, log_paths, suffix=None):
    if args.out:
        if suffix is None:
            return args.out
        root, ext = os.path.splitext(args.out)
        ext = ext or ".png"
        return f"{root}_{suffix}{ext}"
    if len(log_paths) == 1:
        base_dir = os.path.dirname(log_paths[0])
    else:
        base_dir = args.work_dir
    name = "planning_loss.png" if suffix is None else f"planning_loss_{suffix}.png"
    return os.path.join(base_dir, name)


def plot_loss_group(rows, x, keys, iters_per_epoch, log_paths, work_dir, out_path,
                    group_label):
    plt.figure(figsize=(10, 5))
    plotted = []
    for key in keys:
        xs, ys = [], []
        for gx, row in zip(x, rows):
            if key not in row:
                continue
            xs.append(gx)
            ys.append(float(row[key]))
        if not xs:
            continue
        plt.plot(xs, ys, label=key)
        plotted.append(key)
    if not plotted:
        plt.close()
        return False

    epochs = sorted({int(row.get("epoch", 1)) for row in rows})
    for epoch in epochs[1:]:
        boundary = (epoch - 1) * iters_per_epoch
        plt.axvline(boundary, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.xlabel(f"global iter (iters/epoch={iters_per_epoch})")
    plt.ylabel("loss")
    if len(log_paths) == 1:
        title = os.path.basename(log_paths[0])
    else:
        title = f"{len(log_paths)} logs merged ({os.path.basename(work_dir or log_paths[0])})"
    if group_label:
        title = f"{title} — {group_label}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main():
    args = parse_args()
    if args.log:
        log_paths = args.log
    else:
        log_paths = find_all_logs(args.work_dir)

    row_lists = [load_rows(path) for path in log_paths]
    rows = merge_rows(row_lists)
    if not rows:
        raise RuntimeError(f"no training rows found in: {log_paths}")

    iters_per_epoch = args.iters_per_epoch or infer_iters_per_epoch(rows)
    x = [global_step(row, iters_per_epoch) for row in rows]
    epochs = sorted({int(row.get("epoch", 1)) for row in rows})
    work_dir = args.work_dir or (os.path.dirname(log_paths[0]) if log_paths else ".")

    print(f"logs: {len(log_paths)} file(s)")
    for path in log_paths:
        print(f"  - {path}")
    print(f"rows: {len(rows)}  epochs: {epochs}  iters/epoch: {iters_per_epoch}")

    if args.split_reg_cls:
        reg_keys, cls_keys, other_keys = split_reg_cls_keys(args.keys)
        saved = []
        if reg_keys:
            out_reg = resolve_out_path(args, log_paths, suffix="reg")
            if plot_loss_group(rows, x, reg_keys, iters_per_epoch, log_paths,
                               work_dir, out_reg, "reg"):
                saved.append(out_reg)
        if cls_keys:
            out_cls = resolve_out_path(args, log_paths, suffix="cls")
            if plot_loss_group(rows, x, cls_keys, iters_per_epoch, log_paths,
                               work_dir, out_cls, "cls"):
                saved.append(out_cls)
        if other_keys:
            out_other = resolve_out_path(args, log_paths, suffix="other")
            if plot_loss_group(rows, x, other_keys, iters_per_epoch, log_paths,
                               work_dir, out_other, "other"):
                saved.append(out_other)
        if not saved:
            raise RuntimeError(f"none of keys found in {log_paths}: {args.keys}")
        for path in saved:
            print(f"saved: {path}")
        return

    out_path = resolve_out_path(args, log_paths)
    if not plot_loss_group(rows, x, args.keys, iters_per_epoch, log_paths,
                           work_dir, out_path, None):
        raise RuntimeError(f"none of keys found in {log_paths}: {args.keys}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
