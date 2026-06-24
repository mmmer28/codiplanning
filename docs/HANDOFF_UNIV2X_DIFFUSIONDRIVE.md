# UniV2X 环境与研究备忘 → DiffusionDrive 规划迁移

供新开会话 / Agent 接续：已在 UniV2X 跑通协同评测，后续计划学习 DiffusionDrive 并将规划模块迁入。

---

## 环境与仓库状态

- **机器**：`zkyd`，用户目录 `/home/wangaihong`，多卡 3090；数据曾在 **`/data1/v2x_seq`**，注意根分区磁盘空间。
- **Conda 环境**：`univ2x`，**Python 3.8**。
- **核心版本**：**PyTorch 1.9.1+cu111**，**mmcv-full 1.4.0**，**mmdet 2.14.0**，**mmdet3d 0.17.1**；**NumPy ~1.19**（与 MMDet3D 栈兼容）。不要随意整仓再执行会升级 torch/numpy 的 `requirements.txt`。
- **已修复**：`numpy.typing` 仅在 NumPy≥1.20；已将下列文件改为使用 **`np.ndarray`** 注解，避免评测阶段 import 失败：
  - `projects/mmdet3d_plugin/univ2x/dense_heads/motion_head_plugin/motion_optimization.py`
  - `projects/mmdet3d_plugin/univ2x/dense_heads/planning_head_plugin/collision_optimization.py`

---

## 数据与评测链路（SPD → UniV2X）

- SPD 原始数据转换至 **`datasets/V2X-Seq-SPD-New`**（含 cooperative）；**`data/infos/.../spd_infos_temporal_{train,val}.pkl`** 已就绪。
- **协同评测配置**：**`projects/configs_e2e_univ2x/univ2x_coop_e2e.py`**（文档若写 `univ2x_e2e.py` 为笔误）。
- **示例权重**：**`ckpts/univ2x_coop_e2e_stg2.pth`**（协同 Stage2）。
- **分布式评测**：
  ```bash
  ./tools/univ2x_dist_eval.sh <cfg> <ckpt> <gpu_num>
  ```
  完整日志通常 tee 至：**`projects/work_dirs_e2e_univ2x/univ2x_coop_e2e/logs/eval.*`**
- **默认推理输出**：**`output/results.pkl`**（对应 `tools/test.py` 默认 `--out`）。
- **NuScenes 风格中间 JSON**：**`test/univ2x_coop_e2e/<时间戳>/`**（含 `results_nusc.json`、`results_nusc_det.json` 等）。

---

## 指标分成三块（迁移时需分清）

### 1. 检测 / 跟踪 / motion（NuScenes 协议）

日志中的 **`pts_bbox_NuScenes/*`**、`min_ade` / `min_fde` / `miss_rate` 等；与 **自车规划指标不是同一套**。

### 2. 地图 IoU

最终返回字典中的 **`drivable_iou`、`lanes_iou`** 等；**`divider_iou`** 可能出现 **`nan`**（GT 或分母问题）。

### 3. 规划（PlanningHead，与 Diffusion 替换最直接相关）

- 存放在 **`output/results.pkl`** 的 **`planning_results_computed`** 中。
- **键名**：**`L2`、`obj_col`、`obj_box_col`、`obj_out`、`obj_box_out`**。
- **时间轴**：与配置 **`planning_steps=10`**、步长 **0.5s** 一致（列对应约 **0.5s～5.0s**）。
- **含义简述**：
  - **L2**：每个未来步上，预测与 GT 自车轨迹在 **\(x,y\)** 上的误差（带 mask），再在全集本上平均。
  - **`obj_col` / `obj_out`**：用 **轨迹中心点** 与占用图 / 不可行驶图（**`1 - drivable_gt`**）判定。
  - **`obj_box_col` / `obj_box_out`**：用车身边廓 footprint（固定长宽）在 BEV 栅格上判定。
- **终端打印**：`dataset.evaluate()` 在前期会打印 PrettyTable；若日志只看到 NuScenes 段，需往前翻或查 tee 日志。
- **便捷脚本**：**`tools/print_planning_metrics.py`**（支持无 CUDA 时用 CPU 反序列化读 pkl）：
  ```bash
  cd ~/UniV2X && PYTHONPATH=. python tools/print_planning_metrics.py output/results.pkl
  ```

---

## 规划 GT 来源（改 loss / 换 planner 必懂）

- **没有单独「规划标注文件」**。
- **`sdc_planning`** 来自 **`projects/mmdet3d_plugin/datasets/data_utils/spd_trajectory_api.py`** 中的 **`get_sdc_planning_label`**：
  - 沿 NuScenes 风格 **`sample['next']`** 链前进 **`planning_steps`** 次；
  - 将未来帧自车 box 变换到 **当前帧 ego/lidar 坐标系**，取 **\(x, y, yaw)\)**；
  - **`sdc_planning_mask`** 标记有效步；序列结束时后续步无效；
  - **`command`** 由末端横向偏移等启发式规则得到。
- **L2 语义**：每个 **离散步** 上，**replay 得到的真实 pose（在当前 ego 系下）** 与 **模型预测 pose** 对齐比较；无需先插值为连续曲线。

---

## 可视化

- **`tools/univ2x_vis_results.sh`**：
  - 默认 **`PREDROOT=./output/results.pkl`**（与 `test.py` 默认输出一致）。
  - 其他 pkl：**`PREDROOT=/path/to.pkl ./tools/univ2x_vis_results.sh`**

---

## 迁移 DiffusionDrive 时可挂钩的代码位置

- **推理时规划输出**：`projects/mmdet3d_plugin/univ2x/apis/test.py` — `result[0]['planning']['result_planning']['sdc_traj']` → **`PlanningMetric`**。
- **训练与 head**：`univ2x/detectors/univ2x_e2e.py`、`univ2x/dense_heads/planning_head.py` — **`sdc_planning`、`sdc_planning_mask`**。
- **指标实现**：`univ2x/dense_heads/planning_head_plugin/planning_metrics.py` — **`PlanningMetric`**（注意 **`update` 内对轨迹 x 的符号约定**与 BEV 对齐）。
- 替换扩散规划模块时，应保持 **张量形状（如 B×T×C）与坐标约定**一致，并用 **`planning_results_computed` + `print_planning_metrics.py`** 做回归对比。

---

## 新开会话建议粘贴的一段话

> 我在 `/home/wangaihong/UniV2X` 已跑通协同评测；环境 `univ2x`（torch 1.9 / mmcv-full 1.4 / mmdet3d 0.17）；规划指标在 `results.pkl` 的 `planning_results_computed`，可用 `tools/print_planning_metrics.py` 查看；规划 GT 来自 `spd_trajectory_api.get_sdc_planning_label`。现在要集成 DiffusionDrive 的规划模块，详见 `docs/HANDOFF_UNIV2X_DIFFUSIONDRIVE.md`。

---

*文档由会话移交备忘整理，可按实际情况追加 DiffusionDrive 仓库路径与分支。*
