#!/usr/bin/env python
"""Overfit DiffusionPlanningHead on a small subset for sanity check."""
import argparse
import csv
import importlib
import logging
import os

import distutils.version  # noqa: F401 - tensorboard compat on py3.8
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

importlib.import_module("projects.mmdet3d_plugin")
logging.getLogger("mmcv").setLevel(logging.ERROR)


def save_planning_head(model, path, meta=None):
    state_dict = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if k.startswith("planning_head.")
    }
    payload = {
        "state_dict": state_dict,
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_planning_head(model, path):
    payload = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading planning head: {unexpected}")
    return payload.get("meta", {})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    parser.add_argument("--ckpt", default="ckpts/univ2x_coop_e2e_stg2.pth")
    parser.add_argument("--out-dir", default="work_dirs/plan_overfit")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--queue-length", type=int, default=1)
    parser.add_argument(
        "--save-planning-ckpt",
        default="planning_head_overfit.pth",
        help="filename under out-dir; set empty to skip saving",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "需要 GPU"

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "loss.csv")

    cfg = Config.fromfile(args.config)
    cfg.data.train.queue_length = args.queue_length

    print("[1/5] build dataset ...", flush=True)
    dataset = build_dataset(cfg.data.train)
    sample_indices = list(range(min(args.num_samples, len(dataset))))
    print(f"using sample indices: {sample_indices}", flush=True)

    print("[2/5] build model ...", flush=True)
    model = build_model(
        cfg.model_ego_agent,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()

    print(f"[3/5] load checkpoint: {args.ckpt}", flush=True)
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model.cuda()
    print("[3/5] checkpoint loaded", flush=True)

    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("planning_head."):
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["iter", "plan_total", "loss_ade", "loss_diffusion_reg_0", "loss_diffusion_cls_0"]
        )
        csv_file.flush()

        model.train()
        print("[4/5] overfit ...", flush=True)
        for it in range(1, args.iters + 1):
            idx = sample_indices[(it - 1) % len(sample_indices)]
            data = collate([dataset[idx]], samples_per_gpu=1)
            data = scatter(data, [0])[0]

            optimizer.zero_grad(set_to_none=True)
            losses, _ = model(
                return_loss=True,
                other_agent_results=None,
                img_metas=data["ego_agent_data"]["img_metas"],
                **{k: v for k, v in data["ego_agent_data"].items() if k != "img_metas"},
            )

            plan_loss = sum(v for k, v in losses.items() if k.startswith("planning."))
            plan_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 35.0)
            optimizer.step()

            loss_ade = float(losses.get("planning.loss_ade", torch.tensor(0.0)).detach().cpu())
            loss_reg0 = float(
                losses.get("planning.loss_diffusion_reg_0", torch.tensor(0.0)).detach().cpu()
            )
            loss_cls0 = float(
                losses.get("planning.loss_diffusion_cls_0", torch.tensor(0.0)).detach().cpu()
            )
            plan_total = float(plan_loss.detach().cpu())

            csv_writer.writerow([it, plan_total, loss_ade, loss_reg0, loss_cls0])
            csv_file.flush()

            if it == 1 or it % args.log_interval == 0 or it == args.iters:
                print(
                    f"iter={it:03d} | plan_total={plan_total:.4f} | "
                    f"loss_ade={loss_ade:.4f} | loss_diffusion_reg_0={loss_reg0:.4f} | "
                    f"loss_diffusion_cls_0={loss_cls0:.4f}",
                    flush=True,
                )

    print("[5/5] done", flush=True)
    print(f"csv file: {csv_path}", flush=True)

    if args.save_planning_ckpt:
        planning_ckpt_path = os.path.join(args.out_dir, args.save_planning_ckpt)
        save_planning_head(
            model,
            planning_ckpt_path,
            meta={
                "base_ckpt": args.ckpt,
                "num_samples": args.num_samples,
                "iters": args.iters,
                "lr": args.lr,
                "queue_length": args.queue_length,
                "sample_indices": sample_indices,
            },
        )
        print(f"planning head ckpt: {planning_ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
