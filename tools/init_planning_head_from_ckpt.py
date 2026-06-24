"""Copy planning_head weights from a source checkpoint into a target model ckpt.

Typical use: warm-start infrastructure planning with vehicle sub-system weights.

Example:
  python tools/init_planning_head_from_ckpt.py \\
    --src ckpts/univ2x_sub_veh_stg2.pth \\
    --dst ckpts/univ2x_sub_inf_stg2.pth \\
    --out ckpts/univ2x_sub_inf_stg2_with_planning_init.pth
"""

import argparse
import copy

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="checkpoint with planning_head weights")
    parser.add_argument("--dst", required=True, help="base checkpoint to merge into")
    parser.add_argument("--out", required=True, help="output checkpoint path")
    parser.add_argument(
        "--prefix",
        default="model_ego_agent.planning_head.",
        help="parameter prefix to copy from src",
    )
    args = parser.parse_args()

    src = torch.load(args.src, map_location="cpu")
    dst = torch.load(args.dst, map_location="cpu")
    src_state = src.get("state_dict", src)
    dst_state = dst.get("state_dict", dst)

    copied = []
    for key, value in src_state.items():
        if not key.startswith(args.prefix):
            continue
        dst_state[key] = copy.deepcopy(value)
        copied.append(key)

    if not copied:
        raise RuntimeError(
            "No planning_head keys copied. Check --src and --prefix."
        )

    if "state_dict" in dst:
        dst["state_dict"] = dst_state
    else:
        dst = dst_state

    torch.save(dst, args.out)
    print(f"copied {len(copied)} keys to {args.out}")
    for key in copied[:5]:
        print(f"  {key}")
    if len(copied) > 5:
        print("  ...")


if __name__ == "__main__":
    main()
