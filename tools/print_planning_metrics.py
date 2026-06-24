#!/usr/bin/env python
"""从评测输出的 pkl 中打印 planning_results_computed（L2、碰撞/越界相关指标）。"""
import argparse
import io
import pickle
import sys


def _load_pkl_cpu(path):
    import torch

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b), map_location=torch.device("cpu"))
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def _to_float_list(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu()
    if hasattr(v, "tolist"):
        return [float(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


def main():
    parser = argparse.ArgumentParser(
        description="Print UniV2X planning metrics from test output pkl.")
    parser.add_argument(
        "pkl",
        nargs="?",
        default="output/results.pkl",
        help="path to results.pkl (default: output/results.pkl)")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="seconds per planning step for column headers (default: 0.5)")
    args = parser.parse_args()

    try:
        obj = _load_pkl_cpu(args.pkl)
    except Exception as e:
        print(f"Failed to load {args.pkl}: {e}", file=sys.stderr)
        sys.exit(1)

    if "planning_results_computed" not in obj:
        print(
            "No key 'planning_results_computed'. Top-level keys:",
            list(obj.keys()),
            file=sys.stderr,
        )
        print(
            "提示：若评测时未启用 planning_eval / 未走 planning 分支，该键可能不存在。",
            file=sys.stderr,
        )
        sys.exit(1)

    pr = obj["planning_results_computed"]
    preferred_order = [
        "L2",
        "obj_col",
        "obj_box_col",
        "obj_out",
        "obj_box_out",
    ]
    keys = [k for k in preferred_order if k in pr]
    keys.extend(k for k in pr.keys() if k not in keys)

    if not keys:
        print("planning_results_computed is empty.", file=sys.stderr)
        sys.exit(1)

    n = len(_to_float_list(pr[keys[0]]))
    headers = ["metrics"] + [f"{args.dt * (i + 1):.1f}s" for i in range(n)]

    try:
        from prettytable import PrettyTable

        tab = PrettyTable()
        tab.field_names = headers
        for k in keys:
            row = [k] + [f"{x:.4f}" for x in _to_float_list(pr[k])]
            tab.add_row(row)
        print(tab)
    except ImportError:
        colw = max(len(h) for h in headers)
        print(" | ".join(h.ljust(colw) for h in headers))
        sep = "-" * ((colw + 3) * len(headers) - 3)
        print(sep)
        for k in keys:
            vals = _to_float_list(pr[k])
            cells = [k] + [f"{x:.4f}" for x in vals]
            print(" | ".join(c.ljust(colw) for c in cells))

    print()
    print("说明（与 PlanningMetric 对齐）：")
    print("  L2          — 自车规划轨迹与 GT 的 L2（按未来步，带 mask）")
    print("  obj_col     — 占用图上轨迹点落入占用区域的统计（可归为碰撞类指标）")
    print("  obj_box_col — 车Footprint 与占用相交（车身碰撞类）")
    print("  obj_out     — 轨迹点进入「不可行驶」区域")
    print("  obj_box_out — 车身区域进入不可行驶区域")


if __name__ == "__main__":
    main()
