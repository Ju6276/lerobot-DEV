# FastWAM G1 全身遥操作数据集微调指南

本文档面向从 GitHub 克隆本仓库后、需要在自有 G1 全身 loco-manipulation 数据集上微调 FastWAM 的开发者。内容覆盖：**环境配置 → 数据集改造 → G1 混合归一化 → 训练参数 → 启动命令 → 常见问题**。

---

## 目录

1. [概述](#1-概述)
2. [环境配置](#2-环境配置)
3. [数据集准备与修改](#3-数据集准备与修改)
4. [G1 混合归一化](#4-g1-混合归一化)
5. [训练参数说明](#5-训练参数说明)
6. [微调全流程](#6-微调全流程)
7. [启动命令（最终版）](#7-启动命令最终版)
8. [Checkpoint 与推理](#8-checkpoint-与推理)
9. [常见问题排查](#9-常见问题排查)
10. [本仓库相关文件索引](#10-本仓库相关文件索引)

---

## 1. 概述

### 1.1 任务与数据

| 项目 | 说明 |
|------|------|
| 机器人 | Unitree G1 全身 loco-manipulation |
| 示例数据集 | `G1WholebodyLocomotionPickBetweenTablesTeleop-v0` |
| 数据格式 | LeRobot **v3.0** |
| 相机 | 单路 `observation.images.egocentric`（原始 360×640） |
| 状态 | `observation.state`，32 维 |
| 动作 | `action`，36 维 |
| 语言 | `tasks` 中的任务描述 |

### 1.2 本仓库相对上游 LeRobot 的 G1 相关改动

| 改动 | 路径 |
|------|------|
| `states` → `observation.state` 数据集重命名脚本 | `rename_states_to_observation_state.py` |
| G1 混合归一化（min_max + mean_std） | `src/lerobot/policies/fastwam/g1_hybrid_normalization.py` |
| FastWAM 配置项 `g1_hybrid_normalization` | `src/lerobot/policies/fastwam/configuration_fastwam.py` |
| Processor 集成 | `src/lerobot/policies/fastwam/processor_fastwam.py` |

### 1.3 预训练权重来源

FastWAM 默认会加载：

- **动作/视频 DiT 初始化**：`lerobot/fastwam_base`
- **Wan2.2 骨干（VAE、文本编码器等）**：`Wan-AI/Wan2.2-TI2V-5B` / `Wan-AI/Wan2.2-TI2V-5B-Diffusers`

首次训练需能访问 Hugging Face Hub（建议 `hf auth login`）。

---

## 2. 环境配置

### 2.1 系统要求

- **Python** ≥ 3.12  
- **CUDA** GPU（推荐 A100 80GB；FastWAM 显存占用大）  
- **ffmpeg**（视频解码）  
- **Git LFS**（若数据集含 LFS 资源）

### 2.2 克隆仓库

```bash
git clone <你的 GitHub 仓库 URL> lerobot-DEV
cd lerobot-DEV
```

### 2.3 安装依赖（推荐 uv）

```bash
# 安装 uv（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装 FastWAM + 训练依赖
uv sync --locked --extra fastwam --extra training

# 或以可编辑模式安装（开发常用）
uv pip install -e ".[fastwam,training]"
```

**依赖说明：**

| Extra | 包含内容 |
|-------|----------|
| `fastwam` | transformers、diffusers 等 FastWAM 策略依赖 |
| `training` | accelerate、wandb、dataset 相关训练依赖 |

### 2.4 网络与镜像（国内环境）

若 PyPI 官方源极慢，可临时使用国内镜像安装，但**不要单独用 `UV_INDEX_URL` 替换默认源**（可能导致部分包 404）。推荐分步安装：

```bash
# 1. 先从 PyTorch 官方源装 CUDA 版 torch（按需选择 cu128）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. 再装本项目
uv pip install -e ".[fastwam,training]"
```

### 2.5 Hugging Face 登录

```bash
hf auth login
```

用于下载 `Wan-AI/Wan2.2-TI2V-5B`、`lerobot/fastwam_base` 等权重。

### 2.6 验证安装

```bash
uv run python -c "
import torch
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('FastWAMConfig ok:', FastWAMConfig(action_dim=36, proprio_dim=32).type)
"
```

---

## 3. 数据集准备与修改

### 3.1 目录结构（v3.0）

数据集应放在例如 `dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/`：

```
G1WholebodyLocomotionPickBetweenTablesTeleop-v0/
├── meta/
│   ├── info.json          # 特征定义、fps、路径模板
│   ├── stats.json         # 归一化统计量
│   ├── tasks.parquet
│   └── episodes/...
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── observation.images.egocentric/   # 注意相机 key 名
        └── chunk-000/
            └── file-000.mp4
```

`meta/info.json` 中需有：

```json
"codebase_version": "v3.0"
```

### 3.2 将 `states` 永久改为 `observation.state`（方案 B）

FastWAM 读取 **`observation.state`** 作为本体感知，而非旧字段 `states`。

使用仓库根目录脚本（幂等，会自动 `.bak` 备份）：

```bash
uv run python rename_states_to_observation_state.py \
  --dataset-root dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  --fallback-stats dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0_old/meta/stats.json
```

脚本会修改：

1. `meta/info.json`：`features.states` → `features.observation.state`
2. `meta/stats.json`：从 `_old` 备份或 parquet 重算 `observation.state` 统计量
3. `data/**/*.parquet`：列名 `states` → `observation.state`
4. 修复 `observation.prev_height` 等标量/列表不一致（v2.1→v3.0 遗留问题）

仅预览不写盘：

```bash
uv run python rename_states_to_observation_state.py --dry-run
```

### 3.3 修正 `info.json` 中的动态 shape

若 `action` / `observation.state` 的 shape 为 `[-1]`，FastWAM 校验会失败。应改为固定维度：

```json
"action": { "shape": [36], ... }
"observation.state": { "shape": [32], ... }
```

可用 Python 快速修正：

```bash
uv run python -c "
import json
from pathlib import Path
p = Path('dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/meta/info.json')
info = json.loads(p.read_text())
info['features']['action']['shape'] = [36]
info['features']['observation.state']['shape'] = [32]
p.write_text(json.dumps(info, indent=4) + '\n')
print('updated info.json')
"
```

### 3.4 视频路径命名

v3 要求视频目录名与 feature key 一致，例如：

- ✅ `videos/observation.images.egocentric/chunk-000/...`
- ❌ `videos/chunk-000/egocentric/...`（v2.1 旧布局，转换时会报错）

### 3.5 G1 动作/状态维度（modality 参考）

`meta/modality.json`（若存在）定义子向量切片，与混合归一化对应：

**State（32 维，`observation.state`）— 全部 min_max：**

| 子特征 | 维度索引 |
|--------|----------|
| left_hand | 0:7 |
| right_hand | 7:14 |
| left_arm | 14:21 |
| right_arm | 21:28 |
| rpy | 28:31 |
| height | 31:32 |

**Action（36 维）— 混合归一化：**

| 子特征 | 维度索引 | 归一化 |
|--------|----------|--------|
| left/right hand, arm, rpy, height | 0:32 | min_max |
| torso_vx, torso_vy, torso_vyaw, target_yaw | 32:36 | mean_std |

### 3.6 验证数据集可加载

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    repo_id='local/G1WholebodyLocomotionPickBetweenTablesTeleop-v0',
    root='dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0',
)
s = ds[0]
print('observation.state:', s['observation.state'].shape)
print('action:', s['action'].shape)
print('frames:', len(ds))
"
```

期望输出：`observation.state: [32]`，`action: [36]`。

### 3.7 `stats.json` 要求

G1 混合归一化需要 `meta/stats.json` 中包含：

- `observation.state`：`min`, `max`, `mean`, `std`（各 32 维）
- `action`：同上（各 36 维）

若缺少 `observation.state`，运行 `rename_states_to_observation_state.py` 并从 `_old/meta/stats.json` 回填。

---

## 4. G1 混合归一化

### 4.1 为什么需要

原始 G1 训练 pipeline 对关节类维度用 **min_max**（缩放到 [-1,1]），对躯干速度/偏航类维度用 **mean_std**。LeRobot 默认 FastWAM 对整段 `action` / `observation.state` 统一 **MEAN_STD**，与原始设定不一致。

本仓库实现 **G1HybridNormalizerProcessorStep**，在训练 preprocessor 与推理 postprocessor 中按维度切片应用不同公式。

### 4.2 启用方式（CLI）

```bash
--policy.g1_hybrid_normalization=true \
--policy.g1_action_min_max_end=32
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `g1_hybrid_normalization` | `false` | 是否启用 G1 混合归一化 |
| `g1_action_min_max_end` | `32` | action 前 N 维用 min_max，之后用 mean_std |

启用后**无需**再设 `--policy.normalization_mapping`；图像仍为 IDENTITY（不归一化）。

### 4.3 归一化公式

| 模式 | 正向（训练输入） | 统计量 |
|------|------------------|--------|
| **min_max** | `2 * (x - min) / (max - min) - 1` | `min`, `max` |
| **mean_std** | `(x - mean) / (std + eps)` | `mean`, `std` |

推理时 postprocessor 做逆变换。

### 4.4 如何修改归一化逻辑（开发者）

核心文件：`src/lerobot/policies/fastwam/g1_hybrid_normalization.py`

| 类/函数 | 作用 |
|---------|------|
| `G1HybridNormalizationSpec` | 定义 state_dim、action_dim、action_min_max_end |
| `G1HybridNormalizerProcessorStep` | 训练前归一化 state + action |
| `G1HybridUnnormalizerProcessorStep` | 推理后反归一化 action |
| `reconcile_fastwam_g1_processors` | 从 `fastwam_base` 加载 checkpoint 时替换标准 normalizer |

**若你的机器人 action 维度不是 36：**

1. 修改 CLI：`--policy.action_dim=<新维度>`
2. 修改 `--policy.g1_action_min_max_end=<min_max 结束的维度>`（默认 32 表示最后 4 维 mean_std）
3. 更新 `G1HybridNormalizationSpec` 默认值（可选），或仅依赖 CLI 传入的 `proprio_dim` / `action_dim`

**若需完全不同的切片规则**（例如某几维用 QUANTILES）：

1. 在 `g1_hybrid_normalization.py` 的 `_normalize_action_tensor` / `_normalize_state_tensor` 中扩展切片逻辑  
2. 在 `configuration_fastwam.py` 增加配置字段  
3. 在 `processor_fastwam.py` 的 `make_fastwam_pre_post_processors` 中传入新参数  

单元测试：`tests/policies/fastwam/test_g1_hybrid_normalization.py`

```bash
uv run --extra test pytest tests/policies/fastwam/test_g1_hybrid_normalization.py -svv
```

---

## 5. 训练参数说明

### 5.1 数据集参数

| 参数 | 示例 | 说明 |
|------|------|------|
| `--dataset.repo_id` | `local/G1...` | 数据集 ID；`local/` 表示本地集 |
| `--dataset.root` | `./dataset/G1...` | 数据集根目录 |
| `--dataset.use_imagenet_stats` | `false` | **必须 false**（本数据集 stats 无图像 key，且 FastWAM 图像不做 ImageNet 归一化） |

### 5.2 FastWAM 策略参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--policy.type` | `fastwam` | 策略类型 |
| `--policy.action_dim` | `36` | 动作维度，须与数据集一致 |
| `--policy.proprio_dim` | `32` | 本体感知维度；`observation.state` 长度 |
| `--policy.action_horizon` | `32` | 一次预测的动作步数（训练监督长度） |
| `--policy.n_action_steps` | `10` | **仅推理**：每次开环执行步数，须 ≤ horizon |
| `--policy.image_size` | `'[224,448]'` | `(高, 宽)`，**非正方形**；单相机 resize 到 224×448 |
| `--policy.freeze_video_expert` | `true` | 冻结 ~5B 视频专家，只训动作专家 + proprio 编码器 |
| `--policy.use_gradient_checkpointing` | `true` | 用计算换显存 |
| `--policy.torch_dtype` | `bfloat16` | 模型精度 |
| `--policy.device` | `cuda` | 设备 |
| `--policy.push_to_hub` | `false` | 本地微调设 false，避免要求 `policy.repo_id` |
| `--policy.g1_hybrid_normalization` | `true` | 启用 G1 混合归一化 |

**`action_horizon` 约束：** 默认 `num_video_frames=33`、`action_video_freq_ratio=4` 时，horizon 必须是 **8 的倍数**（如 24、32），**不能设为 30**。

**`n_action_steps`：** 训练 loss **不使用**此参数；仅 `select_action` 推理时控制 re-plan 频率。

### 5.3 训练超参

| 参数 | 推荐 | 说明 |
|------|------|------|
| `--batch_size` | `1`～`8` | **每卡** batch；双卡有效 batch = `batch_size × 2` |
| `--steps` | `100000` | 优化器更新步数 |
| `--output_dir` | `./outputs/...` | checkpoint 与日志目录 |
| `--save_freq` | `20000`（默认） | 每 N 步存 checkpoint |

学习率默认 `1e-4`（AdamW），来自 `FastWAMConfig.optimizer_lr`。多卡时 LeRobot **不会**自动缩放学习率，需自行按有效 batch 调整。

### 5.4 图像尺寸说明

`[224,448]` 表示高 224、宽 448：

- **单相机**：整幅图 resize 到 224×448  
- **双相机**：各 224×224，宽度之和为 448  

与源视频 16:9（360×640）不同，由模型内部 resize，无需改数据集分辨率。

---

## 6. 微调全流程

```
克隆仓库 → 安装 [fastwam,training] → hf auth login
    ↓
准备 v3 数据集 → 运行 rename_states_to_observation_state.py
    ↓
修正 info.json shape → 验证 LeRobotDataset 可加载
    ↓
5 step dry-run（单卡）→ 确认无 OOM / 无报错
    ↓
正式训练（单卡或多卡 accelerate launch）
    ↓
outputs/ 下 checkpoint → lerobot-eval / 真机部署
```

### 6.1 Dry-run（5 步，建议先跑）

```bash
cd /path/to/lerobot-DEV
source .venv/bin/activate

lerobot-train \
  --dataset.repo_id=local/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  --dataset.root=./dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  --dataset.use_imagenet_stats=false \
  --policy.type=fastwam \
  --policy.action_dim=36 \
  --policy.proprio_dim=32 \
  --policy.action_horizon=32 \
  --policy.n_action_steps=10 \
  --policy.g1_hybrid_normalization=true \
  --policy.image_size='[224,448]' \
  --policy.freeze_video_expert=true \
  --policy.use_gradient_checkpointing=true \
  --policy.torch_dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --steps=5 \
  --output_dir=./outputs/fastwam_g1_dryrun
```

### 6.2 释放 GPU 显存（若有残留进程）

```bash
nvidia-smi
# 查看占用 PID 后 kill，或清理遗留训练 worker
kill -9 <PID>
```

---

## 7. 启动命令（最终版）

### 7.1 双卡 A100 正式训练（当前推荐配置）

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=local/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  --dataset.root=./dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
  --dataset.use_imagenet_stats=false \
  --policy.type=fastwam \
  --policy.action_dim=36 \
  --policy.proprio_dim=32 \
  --policy.action_horizon=32 \
  --policy.n_action_steps=10 \
  --policy.g1_hybrid_normalization=true \
  --policy.image_size='[224,448]' \
  --policy.freeze_video_expert=true \
  --policy.use_gradient_checkpointing=true \
  --policy.torch_dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=100000 \
  --output_dir=./outputs/fastwam_g1_pick_between_tables
```

**说明：**

- 有效 batch size = `8 × 2 = 16`（每卡 8）  
- 若 OOM，将 `--batch_size` 降为 `4` 或 `1`  
- `accelerate launch` 开头的 warning（`num_machines`、`dynamo_backend`）可忽略  

### 7.2 单卡训练

将上面命令中的 `accelerate launch ... $(which lerobot-train)` 替换为直接调用：

```bash
lerobot-train \
  ...同上参数...
```

### 7.3 后台运行并写日志

```bash
nohup accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  ...参数... \
  > train_fastwam_g1.log 2>&1 &

tail -f train_fastwam_g1.log
```

### 7.4 可选：W&B 日志

```bash
  --wandb.enable=true \
  --wandb.project=fastwam-g1 \
  --job_name=g1_pick_between_tables
```

---

## 8. Checkpoint 与推理

训练产物位于 `--output_dir`，例如：

```
outputs/fastwam_g1_pick_between_tables/
├── checkpoints/
│   └── last/
│       ├── pretrained_model/
│       └── ...
└── ...
```

加载微调后的策略进行推理时，使用 LeRobot 标准 API 或 `lerobot-eval`；推理阶段会：

1. 使用训练时保存的 preprocessor（含 G1 混合归一化 stats）  
2. `select_action` 每次预测 `action_horizon=32` 步，执行前 `n_action_steps=10` 步后 re-plan  

---

## 9. 常见问题排查

| 报错 | 原因 | 解决 |
|------|------|------|
| `'repo_id' argument missing` | 默认 `push_to_hub=true` | `--policy.push_to_hub=false` |
| `KeyError: 'observation.images.egocentric'` | `use_imagenet_stats` 写入不存在的图像 stats | `--dataset.use_imagenet_stats=false` |
| `action feature shape must be (36,), got (-1,)` | `info.json` 中 action shape 为 `[-1]` | 改为 `[36]`，或使用本仓库最新 `FastWAMConfig.set_dataset_feature_metadata` |
| CUDA OOM | batch 过大或视频+Wan 显存高 | 降低 `batch_size`；保持 `freeze_video_expert=true` |
| `action_horizon=30` 校验失败 | horizon 须为 8 的倍数 | 使用 24 或 32 |
| Hub 下载慢 | 网络 | `hf auth login`；配置镜像或预下载权重 |
| GPU 显存未释放 | 遗留训练进程 | `nvidia-smi` + `kill -9 <pid>` |

---

## 10. 本仓库相关文件索引

| 文件 | 用途 |
|------|------|
| `FASTWAM_G1_FINETUNE_GUIDE.md` | 本文档 |
| `rename_states_to_observation_state.py` | 数据集 `states` → `observation.state` |
| `src/lerobot/policies/fastwam/g1_hybrid_normalization.py` | G1 混合归一化实现 |
| `src/lerobot/policies/fastwam/configuration_fastwam.py` | FastWAM + G1 配置项 |
| `src/lerobot/policies/fastwam/processor_fastwam.py` | Pre/post processor 工厂 |
| `docs/source/fastwam.mdx` | 上游 FastWAM 通用文档 |
| `docs/source/multi_gpu_training.mdx` | 多卡训练说明 |
| `tests/policies/fastwam/test_g1_hybrid_normalization.py` | 归一化单元测试 |

---

## 附录：新数据集快速检查清单

- [ ] LeRobot v3.0 目录结构完整  
- [ ] 视频路径 `videos/observation.images.<camera>/...`  
- [ ] 已运行 `rename_states_to_observation_state.py`（若原为 `states`）  
- [ ] `info.json` 中 `action` shape=`[36]`，`observation.state` shape=`[32]`  
- [ ] `stats.json` 含 `observation.state` 与 `action` 的 min/max/mean/std  
- [ ] `lerobot-train` dry-run 5 steps 通过  
- [ ] 正式训练命令含 `g1_hybrid_normalization=true` 与 `use_imagenet_stats=false`  

如有问题，请先跑 dry-run 并对照 [第 9 节](#9-常见问题排查)。
