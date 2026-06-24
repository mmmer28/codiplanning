#!/bin/bash
# 默认与 tools/test.py 的 --out output/results.pkl 一致。
# 若用车端专用结果： PREDROOT=./output/vehicle_results.pkl ./tools/univ2x_vis_results.sh
export PYTHONPATH=$PYTHONPATH:./
PREDROOT="${PREDROOT:-./output/results.pkl}"
python ./tools/analysis_tools/visualize/univ2x_run.py \
    --predroot "${PREDROOT}" \
    --out_folder ./output_visualize \
    --demo_video test_demo.avi \
    --project_to_cam 0 \
    --dataroot datasets/V2X-Seq-SPD-New/cooperative