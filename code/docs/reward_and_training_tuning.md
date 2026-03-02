# 不加 Turn Reward 时的优化方向

在不引入 per-turn reward 的前提下，从**结果型 reward 常数/权重**和**训练超参**两方面可做如下调整。

---

## 训练脚本与 2 卡配置

| 脚本 | 卡数 | 说明 |
|------|------|------|
| **train_clarify_2gpu.sh** | 2 | 主训练：Few-Shot + adaptive_clarify，答错不罚，lr=5e-7、warmup=0.2 |
| **train_clarify_consistency_2gpu.sh** | 2 | 同上 + **self-consistency**（consistency=0.2） |
| **train_clarify_full_2gpu.sh** | 2 | **四分量全开**：F1 + adaptive_clarify + confidence=0.1 + consistency=0.2 |
| **train_fewshot_clarify.sh** | 2 | 同上数据/奖励、无 consistency，lr=1e-6、warmup=0.15 |
| train_ablation_adaptive_clarify_only.sh | 2 | 消融：仅 adaptive_clarify |
| train_ablation_confidence_only.sh | 2 | 消融：仅 confidence |
| train_ablation_confidence_plus_clarify.sh | 2 | 消融：confidence + adaptive_clarify |

以上脚本均已统一为 **#SBATCH --gpus=2** 与 **trainer.n_gpus_per_node=2**，**trainer.total_training_steps=500**（原 200）。提交示例：`sbatch train_clarify_2gpu.sh`。

---

## 1. 结果型 Adaptive Clarify Reward 可调项

当前逻辑（`main_ppo.py`）：

- `clarify=0, F1≥0.5` → +0.2（效率 bonus）
- `clarify>0, F1≥0.5` → +0.15×min(clarify_cnt, 3)
- `clarify>0, F1<0.5` → **-0.08×min(clarify_cnt, 4)**
- `clarify=0, F1<0.5` → 0

现象：clarify 率仍会在训练中塌到 0，说明「clarify 但答错」被罚后，在 GRPO 的 advantage 归一化下，不 clarify 的轨迹更占优。

### 1.1 答错时不罚 clarify（推荐先做）

- **做法**：`CLARIFY_WRONG_PENALTY_PER_TURN = 0`
- **效果**：「clarify 但答错」= 0，与「不 clarify 答错」一致，模型不会因答错而被惩罚 clarify 行为本身，只少拿 F1。
- **风险**：可能多出无效 clarify，但先保住 clarify 率再靠后续数据/规则收。

### 1.2 降低「不 clarify 且答对」的吸引力

- **做法**：`NO_CLARIFY_CORRECT_BONUS` 从 0.2 降到 0.1 或 0.05
- **效果**：让「clarify 且答对」相对更优（例如 0.15×2=0.3 vs 0.1），减少策略过早收敛到「一律不问」。

### 1.3 提高 clarify 分量权重

- **做法**：脚本里 `+reward_weights.adaptive_clarify=0.5` 或 0.6（当前 0.3）
- **效果**：clarify 相关 reward 在总 reward 里占比更大，梯度更明显。
- **注意**：过大可能压过 F1，需看 val F1/clarify 平衡。

### 1.4 提高「clarify 且答对」的 per-turn 奖励

- **做法**：`CLARIFY_RIGHT_BONUS_PER_TURN` 从 0.15 提到 0.2
- **效果**：clarify 且答对时总 bonus 更高（如 3 turn → 0.6），相对「不 clarify +0.2」更有优势。

---

## 2. 训练超参可调项

| 参数 | 当前典型值 | 可调方向 | 目的 |
|------|------------|----------|------|
| **actor lr** | 1e-6 | 降到 5e-7 | 更新更稳，减少 clarify 快速塌陷 |
| **lr_warmup_steps_ratio** | 0.15 | 提到 0.2~0.25 | 前段更保守，利于维持 clarify |
| **kl_loss_coef** | 0.03 | 保持或略提到 0.04 | 别偏离 ref 太快，ref 已有 clarify 能力 |
| **entropy_coeff** | 0.001 | 提到 0.002~0.003 | 略增探索，多尝试 clarify |
| **temperature** (rollout) | 1 | 保持或略升 1.05 | 采样稍多样，避免过早确定「不问」 |
| **total_training_steps** | 200 | 可适当加长 | 给策略更多步数收敛到「该问则问」 |

不建议为保 clarify 把 KL 系数调得过大（如 >0.05），否则几乎不更新；也不建议 lr 再明显增大，已有 KL 爆炸前科。

---

## 3. 建议的优先顺序

1. **先改 reward**：`CLARIFY_WRONG_PENALTY_PER_TURN = 0`（答错不罚 clarify）。
2. 若 clarify 率仍掉得快：再试 `NO_CLARIFY_CORRECT_BONUS = 0.1` 或 `adaptive_clarify` 权重 0.5。
3. 再考虑训练参数：actor lr 5e-7、warmup 0.2、entropy_coeff 0.002。

若你坚持完全不加 turn-level reward，上述 1+2 通常能明显缓解 clarify 塌陷；若仍不够，再考虑加 per-turn 小 bonus 或 curriculum（先多给 clarify 数据/奖励，再收紧）。
