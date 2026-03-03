# Clarify-R1 实验运行指南

> 给服务器 AI 的操作手册。按顺序执行即可。

---

## 概述

本实验对比 4 种 reward mode 训练 proactive clarification 模型，验证 IPO (Information Gain Policy Optimization) 相比 baseline 的优势。

**Base Model:** Qwen2.5-3B-Instruct
**Dataset:** AmbigNQ (ambiguous QA)
**Framework:** verl (GRPO-based RL)
**GPU:** 4× GPU (SLURM cluster)

### 实验矩阵

| # | Reward Mode | 脚本 | 核心逻辑 | 预期结果 |
|---|------------|------|---------|---------|
| 0 | SFT Warmup | `train_sft_clarify.sh` | 有监督微调：教模型 XML 格式 | 格式正确率 ~95%+ |
| 1 | Outcome-Only | `train_rl_outcome_only.sh` | 只有最终 F1 | 会 collapse（baseline下界）|
| 2 | Equalized | `train_rl_equalized.sh` | F1 + 每轮固定 +0.1 | 稳定但过度 clarify |
| 3 | **IPO** | `train_rl_ipo.sh` | F1 + IG turn reward | **目标最好** |
| 4 | Adaptive | `train_rl_adaptive.sh` | 4 成分加权 | 复杂但效果中等 |

---

## 前置条件

### 环境
```bash
conda activate searchr1
```

### 依赖服务（必须在跑 RL 之前启动）
1. **Retrieval Server** — `http://10.10.211.118:8000/retrieve`（dense passage retrieval，为 `<search>` action 提供检索结果）
2. **GPT-4 Simulator** — `http://10.10.211.118:8001`（模拟用户回复 `<clarify>` 问题）

如果这两个服务没跑，RL 训练的 rollout 会卡住或报错。

### 服务启动方式（如需手动启动）
```bash
# Retrieval server (如果没在跑)
# 检查: curl http://10.10.211.118:8000/retrieve -X POST -d '{"query": "test", "topk": 3}'

# GPT-4 simulator (如果没在跑)
# 检查: curl http://10.10.211.118:8001/batch_generate -X POST -H "Content-Type: application/json" -d '{"prompts": ["test"]}'
```

### 数据
RL 训练数据已有：`scripts/data_process/data/ambignq_fewshot/train.parquet`
SFT 训练数据会自动生成（见 Step 1）。

---

## Step 1: SFT Warmup（必须先跑）

### 为什么需要 SFT？
Base model 的 XML tag 格式正确率只有 ~50%。不先 SFT 的话，RL 训练时大部分 rollout 解析不出 answer，reward 全是 0，训练直接 collapse。

### 运行
```bash
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new/code
sbatch train_sft_clarify.sh
```

### 内部流程
1. 自动检测 `scripts/data_process/data/sft_clarify/train.parquet` 是否存在
2. 如果不存在，自动运行 `sft_clarify_data.py` 生成数据（从 AmbigNQ 构造 50/50 ambiguous/non-ambiguous 的 gold 轨迹）
3. 用 FSDP 在 4 GPU 上训练 4 个 epoch
4. 自动创建 symlink: `verl_checkpoints/sft-clarify-warmup/global_step_final` → 最新 checkpoint

### SFT 数据格式
- **Ambiguous 题**: prompt → `<think>This question is ambiguous...</think>\n<clarify>Are you asking: {disambiguated_question}</clarify>`
- **Non-ambiguous 题**: prompt → `<think>This question is clear...</think>\n<answer>{gold_answer}</answer>`
- 只训练**第一轮**模型输出（不包括 `<user_response>` 和后续回答）

### 预计时间
~2-4 小时（4 GPU）

### 验证 SFT 完成
```bash
ls verl_checkpoints/sft-clarify-warmup/global_step_final/
# 应该能看到 model 文件（config.json, model-*.safetensors, tokenizer 等）
```

---

## Step 2: 启动 4 组 RL 实验

SFT 完成后，**并行**提交 4 个 RL 训练：

```bash
sbatch train_rl_outcome_only.sh   # Job 1: Outcome-Only baseline
sbatch train_rl_equalized.sh      # Job 2: Equalized (UserRL-style)
sbatch train_rl_ipo.sh            # Job 3: IPO (core innovation)
sbatch train_rl_adaptive.sh       # Job 4: Adaptive (existing complex)
```

> 如果 GPU 不够同时跑 4 个（每个需要 4 GPU），就按优先级依次跑：
> **IPO > Outcome-Only > Equalized > Adaptive**

### ⚠️ 重要：确认 BASE_MODEL 路径

所有 RL 脚本默认引用 `verl_checkpoints/sft-clarify-warmup/global_step_final`。如果 SFT 的 symlink 创建失败，需要手动修改每个 RL 脚本里的 `BASE_MODEL`：

```bash
# 找到实际 checkpoint 路径
ls -td verl_checkpoints/sft-clarify-warmup/global_step_* | head -1
# 例如: verl_checkpoints/sft-clarify-warmup/global_step_400

# 然后修改每个 RL 脚本
sed -i "s|global_step_final|global_step_400|" train_rl_*.sh
```

### 训练参数（所有 RL 脚本共用）

| 参数 | 值 | 说明 |
|------|---|------|
| `total_training_steps` | 500 | 总训练步数 |
| `train_batch_size` | 32 | 每步 32 个 query |
| `n_agent` | 5 | 每个 query 5 个 rollout |
| `max_turns` | 4 | 最多 4 轮对话 |
| `max_clarify_turns` | 3 | 最多 3 次 clarify |
| `lr` | 5e-7 | 学习率（比之前 1e-6 更保守）|
| `kl_loss_coef` | 0.03 | KL 惩罚（比之前 0.001 强 30x）|
| `grad_clip` | 1.0 | 梯度裁剪 |
| `save_freq` | 50 | 每 50 步保存 checkpoint |
| `test_freq` | 10 | 每 10 步跑 validation |

### 各 reward mode 的核心区别

**Outcome-Only** (`main_ppo_outcome_only.py`):
```
reward[last_token] = F1(predicted_answer, gold_answers)
# 中间轮全是 0 → 预期会 collapse
```

**Equalized** (`main_ppo_equalized.py`):
```
reward[last_token] = F1
reward[each </clarify>] += 0.1    # 固定奖励，不管 clarify 有没有用
```

**IPO** (`main_ppo_ipo.py`):
```
reward[last_token] = F1
reward[each </clarify> or </search>] += alpha * F1 / n_intermediate
# alpha=0.5 (可通过 +ipo.alpha=X 调整)
# 关键：只有最终答对了，中间步才有 reward（IG proxy）
```

**Adaptive** (`main_ppo.py`, 已有):
```
R = 1.0*F1 + 0.3*adaptive_clarify + 0.1*confidence + 0.2*consistency
```

### 预计时间
每个实验 ~12-24 小时（4 GPU, 500 steps）

---

## Step 3: 监控训练

### 看日志
```bash
# 实时看某个 job 的输出
tail -f rl-ipo.log
# 或
tail -f out/<job_id>-rl-ipo.out
```

### 关键指标
在日志里搜索这些关键词：

```bash
# 看 reward 趋势（应该逐步上升）
grep "reward/mean" rl-ipo.log

# 看 F1 趋势
grep "F1=" rl-ipo.log | tail -20

# 看 clarify 比例（IPO 应该 25-40%，Outcome-Only 会降到 0%）
grep "clarify" rl-ipo.log | tail -20

# 看是否有 NaN（训练崩溃的信号）
grep -i "nan" rl-ipo.log
```

### 训练 collapse 的信号
- reward/mean 持续下降或变成 0
- clarify rate 降到 0%（Outcome-Only 预期如此，其他不应该）
- 出现 NaN
- loss 突然跳到很大的值

如果 IPO 或 Equalized collapse，可能需要：
1. 进一步降 lr（试 2e-7）
2. 增大 KL coef（试 0.05）
3. 增大 warmup ratio（试 0.3）

---

## Step 4: 评估

训练完成后，对每个实验的 checkpoint 跑评估：

```bash
# 评估 IPO 最终 checkpoint
python -m verl.trainer.main_eval \
    --model_path verl_checkpoints/rl-ipo/global_step_500 \
    --data_path scripts/data_process/data/ambignq_fewshot/tinytest.parquet \
    --output_path results/eval_ipo.json

# 类似地评估其他 3 个
```

### 需要收集的指标

| Metric | 定义 | Paper 用途 |
|--------|------|-----------|
| Answer F1 | token-level F1 vs gold answers | 主要结果 |
| Clarify Rate | 使用了 clarify 的 query 比例 | 衡量是否过度/不足 clarify |
| Search Rate | 使用了 search 的 query 比例 | 衡量检索使用 |
| Turns/Query | 平均对话轮数 | 效率 |
| Format Error Rate | 无法解析 answer 的比例 | SFT 有效性 |

### 目标结果表（Paper Table）

| Reward Mode | F1 ↑ | Clarify Rate | Over-Clarify ↓ | Collapse? |
|-------------|------|-------------|----------------|-----------|
| Outcome-Only | ~0.20 | ~0% | N/A | Yes |
| Equalized | ~0.35 | ~40% | High | No |
| **IPO** | **~0.45** | **~30%** | **Low** | **No** |
| Adaptive | ~0.40 | ~25% | Medium | No |

---

## 文件清单

### 新文件（需要 push）

| 路径 | 行数 | 作用 |
|------|-----|------|
| `verl/trainer/reward_utils.py` | 216 | 共享 base class：F1 计算、文本提取、turn boundary 检测 |
| `verl/trainer/main_ppo_outcome_only.py` | 185 | Outcome-Only reward manager + training entry point |
| `verl/trainer/main_ppo_equalized.py` | 203 | Equalized reward manager + training entry point |
| `verl/trainer/main_ppo_ipo.py` | 232 | **IPO reward manager** + training entry point |
| `scripts/data_process/sft_clarify_data.py` | 293 | SFT 训练数据生成（从 AmbigNQ 构造 gold 轨迹）|
| `train_sft_clarify.sh` | 97 | SFT warmup 训练脚本（SLURM）|
| `train_rl_outcome_only.sh` | 123 | Outcome-Only RL 训练脚本 |
| `train_rl_equalized.sh` | 122 | Equalized RL 训练脚本 |
| `train_rl_ipo.sh` | 123 | IPO RL 训练脚本 |
| `train_rl_adaptive.sh` | 126 | Adaptive RL 训练脚本（+ stability fixes）|
| `docs/reward_modes_report.md` | 207 | 技术文档 |
| `docs/EXPERIMENT_GUIDE.md` | 本文件 | 实验运行指南 |

### 已有文件（不需要改）

| 路径 | 作用 |
|------|------|
| `verl/trainer/main_ppo.py` | 原有 Adaptive reward manager（被 `train_rl_adaptive.sh` 直接调用）|
| `verl/trainer/fsdp_sft_trainer.py` | SFT 训练器 |
| `verl/trainer/ppo/ray_trainer.py` | PPO 训练编排 |
| `verl/trainer/ppo/core_algos.py` | PPO 核心算法 |
| `scripts/data_process/ambignq_fewshot.py` | RL 训练数据（已有）|

---

## 已知问题 & 注意事项

### 1. Token 位置近似
`reward_utils.py` 的 `find_turn_token_positions()` 通过 re-encode prefix text 来定位 closing tag 的 token index。由于 BPE tokenizer 的 decode→re-encode 不完全可逆，位置可能偏差 1-2 token。对 RL 训练影响不大，但如果观察到 reward 分配异常，这是一个排查方向。

### 2. IPO alpha 调参
默认 `alpha=0.5`，意味着一个成功的 clarify 轨迹的总 reward = `F1 * (1 + 0.5) = 1.5 * F1`，而直接回答只有 `F1`。如果发现模型过度 clarify，降低 alpha（试 0.3）；如果 clarify 不够，增大 alpha（试 0.8）。

修改方式：在 `train_rl_ipo.sh` 最后加 `+ipo.alpha=0.3`。

### 3. SFTDataset 依赖
`fsdp_sft_trainer.py` 引用的 `SFTDataset` 来自 pip 安装的 verl 包，不在本地代码里。确保服务器上 `pip install verl` 过。

### 4. Azure API Key
脚本里硬编码了 Azure OpenAI API key。如果 key 过期或配额用完，GPT-4 simulator 会挂。可以在脚本里换成新 key，或者用本地模型替代 simulator。

---

## 快速开始（TL;DR）

```bash
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new/code

# 1. SFT (必须先跑完)
sbatch train_sft_clarify.sh
# 等完成后验证
ls verl_checkpoints/sft-clarify-warmup/global_step_final/

# 2. RL (SFT 完成后)
sbatch train_rl_ipo.sh            # 最重要
sbatch train_rl_outcome_only.sh   # baseline
sbatch train_rl_equalized.sh      # 对比
sbatch train_rl_adaptive.sh       # 对比

# 3. 监控
tail -f rl-ipo.log
```
