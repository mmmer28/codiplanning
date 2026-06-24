# UniV2X Diffusion 规划头迁移与训练 — 工作移交备忘

供下一个 Agent / 会话接续。仓库根目录：`/home/wangaihong/UniV2X`。

---

## 1. 总体目标（做了什么）

在 **UniV2X 协同 E2E 框架**上，将 **DiffusionDrive 风格的 Diffusion 规划头**接入 ego 车 `planning_head`，在 **冻结感知 / 地图 / 运动 / 占用** 的前提下，**只训练 `model_ego_agent.planning_head`**，完成数据准备、训练、评测闭环。

**不是**重写整个 UniV2X，而是：

1. 新增 `DiffusionPlanningHead` 并替换 `univ2x_coop_e2e.py` 中的原 `PlanningHeadSingleMode`；
2. 修正 scheduler、坐标归一化、位置编码、loss 目标等与 DiffusionDrive 对齐的关键差异；
3. 在 SPD cooperative **train 全量**上 fine-tune 20 epoch；
4. 在 **val** 上评测规划指标（L2 / 碰撞 / 越界），并与原生 stg2 基线对比。

---

## 2. 环境与依赖（勿随意升级）

| 项 | 版本 / 说明 |
|----|-------------|
| Conda | `univ2x`，Python 3.8 |
| PyTorch | 1.9.1+cu111 |
| mmcv-full | 1.4.0 |
| mmdet / mmdet3d | 2.14.0 / 0.17.1 |
| **diffusers** | **0.14.0**（必须；0.36+ 与 torch1.9 不兼容） |
| **huggingface-hub** | **0.13.4** |
| NumPy | ~1.19（`numpy.typing` 已改为 `np.ndarray` 注解，见下） |

曾长期误用 `_FallbackDDIMScheduler`（线性噪声，非真实 DDIM），是早期 **reg 难降** 的主要原因之一；修 diffusers 后应确认日志里能 `import DDIMScheduler` 成功。

---

## 3. 代码改动清单

### 3.1 核心：Diffusion 规划头（新增）

**文件**：`projects/mmdet3d_plugin/univ2x/dense_heads/diffusion_planning_head.py`

- 类 `DiffusionPlanningHead(PlanningHeadSingleMode)`，已在 `dense_heads/__init__.py` 注册。
- **条件输入**：冻结分支的 `planning_sdc_embedding` / track query、BEV、command、ego_status、agent queries。
- **Diffusion**：`DDIMScheduler`（`prediction_type='sample'`），训练随机 timestep，推理 `inference_steps=2`（配置可调）。
- **Anchor**：`data/plan_anchors/spd_plan_anchors_20x10.npy`（20 modes × 10 steps × xy）。
- **Loss**（与 DiffusionDrive 对齐的训练目标）：
  - `loss_diffusion_cls_*`：focal，权重 10.0；
  - `loss_diffusion_reg_*`：L1 on best anchor，权重 **8.0**；
  - `train_aux_planning_losses=False` 时 **关闭** `loss_ade` / `loss_collision_*`。
- **归一化**（关键修复）：
  - `_norm_xy` / `_denorm_xy`：**min-max affine → [-1, 1]**，支持配置 `xy_norm_bounds`；
  - 训练配置使用 GT 统计：`[[-4.0328, -17.6693], [54.2865, 20.0801]]`（train p1–p99 + margin，见 `tools/scan_planning_norm_bounds.py`）。
- **位置编码**（关键修复）：`_decode()` 中对 **米制** `noisy_traj_points` 做 `gen_sineembed_for_position`，**不再**先 `_norm_xy` 再编码。

**与 DiffusionDrive 的已知差异（可接受）**：

| 项 | DiffusionDrive | 本实现 |
|----|----------------|--------|
| 轨迹维度 | x,y,heading | 仅 **xy** |
| 步数 | 8 | **10**（0.5s 间隔 → 5s） |
| 条件 | Transfuser BEV | 冻结 UniV2X track/motion BEV + command/ego_status |
| norm 范围 | NAVSIM 硬编码 | **SPD GT 统计** |

### 3.2 主配置切换

**文件**：`projects/configs_e2e_univ2x/univ2x_coop_e2e.py`

- `planning_head.type = 'DiffusionPlanningHead'`
- `planning_steps = 10`，`planning_eval=True`
- `plan_anchor_path`、`xy_norm_bounds`（默认与 fulltrain 一致）
- `data_root = datasets/V2X-Seq-SPD-New/cooperative/`
- `data.test` 与 `data.val` 均指向 **`spd_infos_temporal_val.pkl`**

### 3.3 规划头专用训练配置（新增）

**文件**：`projects/configs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain.py`

- 只训 `model_ego_agent.planning_head.*`（`TrainableParamPrefixHook`）
- `train_aux_planning_losses=False`
- `optimizer`: AdamW lr=5e-5，`grad_clip max_norm=35`
- `total_epochs=20`，`checkpoint_config=None`
- `work_dir = projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain`

### 3.4 评测配置（新增）

**文件**：`projects/configs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_eval.py`

- 继承 fulltrain，去掉 `resume_from` / 训练 hooks。

### 3.5 训练 Hooks（扩展）

**文件**：`projects/mmdet3d_plugin/univ2x/hooks/custom_hooks.py`

| Hook | 作用 |
|------|------|
| `TrainableParamPrefixHook` | 每 iter 强制仅 planning_head 可训，其余 `eval()` |
| `BestLatestPlanningCheckpointHook` | 每 epoch 写 `latest.pth`，按 reg 更新 `best.pth`，删 `epoch_*.pth` |

**已知问题**：`best.pth` 在整轮训练中 **未成功按 epoch 更新**（日志无 `Updated best`）。`best.pth` 仍停留在手动从 epoch3 初始化的权重；**请用 `latest.pth`（epoch 20）做推理/续训**。疑似 `after_train_epoch` 时 `log_buffer.output` 的 key 与 `planning.loss_diffusion_reg_*` 对不上，需后续修 hook。

### 3.6 工具脚本

| 文件 | 用途 |
|------|------|
| `tools/scan_planning_norm_bounds.py` | 扫 train GT，输出建议 `xy_norm_bounds` JSON |
| `tools/plot_training_loss.py` | 从 `.log.json` 画 loss；支持 `--split-reg-cls` 分开 reg/cls |
| `tools/print_planning_metrics.py` | 从 `results.pkl` 打印 L2 / obj_col 等 |

### 3.7 其它小修

- `motion_optimization.py` / `collision_optimization.py`：`np.ndarray` 替代 `numpy.typing`（评测 import 兼容 py3.8 + numpy1.19）。

---

## 4. 数据与路径

- **数据不在 git 内**（`.gitignore: datasets`），按 `docs/DATA_PREP.md` 本地准备。
- 当前使用：`datasets/V2X-Seq-SPD-New/cooperative/`
- 大图/点云为 **软链** → `/data1/v2x_seq/V2X-Seq-SPD/`（约 41G）；`datasets/` 本地约 2.1G 元数据。
- **索引**：`data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_train.pkl`（**1521** 条）/ `val.pkl`（**675** 条）。
- 规划 GT：`spd_trajectory_api.get_sdc_planning_label`（沿 sample 链未来 10 步 ego xy/yaw）。

---

## 5. 训练流程与命令

### 5.1 启动训练（4 卡）

```bash
cd ~/UniV2X && conda activate univ2x

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/univ2x_dist_train.sh \
  ./projects/configs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain.py 4
```

- `univ2x_dist_train.sh` 会把 `--work-dir` 设为 `projects/work_dirs_e2e_univ2x/<config名>/`（与配置里 `work_dir` 一致即可）。
- **续训用 `resume_from`**（恢复 optimizer/epoch），路径必须是 **相对仓库根目录的完整路径**，不能只写 `epoch_3.pth`。
- **不要用 `load_from`** 做续训（会重置 optimizer）。

### 5.2 实际训练时间线

| 阶段 | 日志 | 说明 |
|------|------|------|
| Run A | `20260528_132310.log` | epoch 1–4，DDIM+新 norm；epoch4 末 reg₀≈9.1 |
| Run B | `20260529_014752.log` | **失败**：`resume_from=epoch_3.pth` 路径错，等价从头训 epoch1 |
| Run C | `20260529_015636.log` | **正式续训**：`resumed epoch 3` → 训到 **epoch 20** |

### 5.3 Checkpoint 产物

目录：`projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/`

| 文件 | 说明 |
|------|------|
| **`latest.pth`** | **epoch 20，推荐用于评测/续训** |
| `best.pth` | ⚠️ 仍为 epoch3 占位，勿用 |
| `results.pkl` | val 评测输出（62MB） |
| `planning_loss_reg.png` / `planning_loss_cls.png` | loss 曲线 |

---

## 6. 训练结果（planning loss，train iter 均值）

续训段 epoch 4→20（`20260529_015636.log.json`）：

| Epoch | reg₀+reg₁ | reg₀ | cls₀ | grad_norm(均值) |
|-------|-----------|------|------|------------------|
| 4 | 19.06 | 9.58 | 0.125 | 50.4 |
| 10 | 16.17 | 8.24 | 0.114 | 71.0 |
| 15 | 14.98 | 7.63 | 0.105 | 81.4 |
| 20 | **13.88** | **7.07** | **0.104** | 84.9 |

- 相对 epoch4，reg 总和约 **↓27%**；epoch 18–20 降幅放缓，**未完全平台**但仍在缓降。
- `grad_norm` 升至 ~85 属预期（reg 权重大 + clip=35，日志为 **裁剪前** 范数）；reg 仍在降则不必 panic。
- 画完整 1–20 epoch 曲线需 **合并两个 log**（勿含失败的 `014752`）：

```bash
python tools/plot_training_loss.py \
  --log projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/20260528_132310.log.json \
         projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/20260529_015636.log.json \
  --split-reg-cls \
  --keys planning.loss_diffusion_reg_0 planning.loss_diffusion_reg_1 \
         planning.loss_diffusion_cls_0 planning.loss_diffusion_cls_1 \
  --out projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/planning_loss.png
```

---

## 7. 评测结果（val，`latest.pth`）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/univ2x_dist_eval.sh \
  ./projects/configs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_eval.py \
  ./projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/latest.pth \
  4 \
  --out ./projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/results.pkl

PYTHONPATH=. python tools/print_planning_metrics.py \
  projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/results.pkl
```

### 7.1 Diffusion（epoch20）vs 原生 stg2

| 指标 | 原生 stg2<br>`output/results_yuanshi1.pkl` | Diffusion latest<br>`fulltrain/results.pkl` |
|------|---------------------------------------------|---------------------------------------------|
| 规划头 | `PlanningHeadSingleMode` | `DiffusionPlanningHead`（已训练） |
| 权重 | `ckpts/univ2x_coop_e2e_stg2.pth` | `latest.pth` |
| L2 @0.5s | 1.09 m | **0.78 m** |
| L2 @2.0s | 2.19 m | **1.97 m** |
| L2 @5.0s | **5.25 m** | 5.93 m |
| obj_box_out @5s | 5.0% | 4.7% |

**解读**：短期 L2 明显改善；**5s 长期略差于原生 stg2**；碰撞/越界率均较低。det/map/track 指标与冻结基座一致，**不看 det mAP 判断规划**。

评测日志：`projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_eval/logs/eval.05310905`

---

## 8. 常见坑

1. **`resume_from` 路径**：必须 `projects/work_dirs_e2e_univ2x/.../epoch_3.pth`，不能仅 `epoch_3.pth`。
2. **`load_from` vs `resume_from`**：改 loss 后续训只用 `resume_from`；`load_from` 重置 optimizer。
3. **磁盘**：每个 ckpt ~1.1G；曾 100% 满导致 `epoch_4.pth` 损坏；用 hook 只留 latest+best。
4. **勿混用旧目录**：`work_dirs/diffusion_plan_fulltrain/` 为更早实验；当前正式目录为 `projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/`。
5. **stg2 评测不是 diffusion**：`results_yuanshi1.pkl` 是原版规划头；把 stg2 载入当前 `DiffusionPlanningHead` 配置会缺 ~109 个 diffusion 参数。
6. **`workers_per_gpu`**：4×8 易 OOM worker，已改为 2。

---

## 9. 建议的下一步

1. **修 `BestLatestPlanningCheckpointHook`**：确认 epoch 末 `log_buffer` 的 key（是否带 `planning.` 前缀），使 `best.pth` 真正跟踪最优 reg；或直接把 `latest.pth` 拷为 `best.pth`。
2. **续训 10–20 epoch**（lr 减半），盯 reg 是否平台 & val L2@5s 是否低于 5.25m。
3. **与 stg2 公平对比**：同一 eval 配置、同一 val；可选调 `inference_steps`、collision 推理后处理。
4. **可选**：降 `loss_diffusion_reg_weight`（8→4~6）或 `max_norm`（35→25）缓解 grad_norm 偏大。

---

## 10. 开新会话可粘贴的摘要

> 在 `/home/wangaihong/UniV2X` 已完成 DiffusionDrive 风格 `DiffusionPlanningHead` 接入 UniV2X 协同 E2E：只训 planning_head，20 epoch，产物 `projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_diffusion_plan_fulltrain/latest.pth`；val 规划 L2@0.5s=0.78、@5s=5.93（原生 stg2：1.09/5.25）。详见 `docs/HANDOFF_DIFFUSION_PLANNING_WORK.md` 与 `docs/HANDOFF_UNIV2X_DIFFUSIONDRIVE.md`。

---

*文档生成：Diffusion 规划迁移训练/评测会话移交。*
