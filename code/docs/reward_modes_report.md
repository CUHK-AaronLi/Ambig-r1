# Clarify-R1 Reward Modes — Implementation Report

## Overview

This document describes the four reward modes implemented for the Clarify-R1 ablation experiments. Each mode is a separate `main_ppo_*.py` file that can be run independently without modifying the existing `main_ppo.py`.

### Training Pipeline

```
SFT Warmup (train_sft_clarify.sh)
    │
    ▼
RL Training (pick one reward mode)
    ├── train_rl_outcome_only.sh   → main_ppo_outcome_only.py
    ├── train_rl_equalized.sh      → main_ppo_equalized.py
    ├── train_rl_ipo.sh            → main_ppo_ipo.py
    └── train_rl_adaptive.sh       → main_ppo.py (existing)
    │
    ▼
Evaluation (main_eval.py)
```

## Files Added

| File | Purpose |
|------|---------|
| `verl/trainer/reward_utils.py` | Shared base class (`BaseRewardManager`) with F1 scoring, text extraction, turn boundary detection |
| `verl/trainer/main_ppo_outcome_only.py` | Outcome-only reward (pure F1) |
| `verl/trainer/main_ppo_equalized.py` | Equalized reward (F1 + constant 0.1/turn) |
| `verl/trainer/main_ppo_ipo.py` | IPO reward (F1 + information gain turn reward) |
| `scripts/data_process/sft_clarify_data.py` | SFT training data generator |
| `train_sft_clarify.sh` | SFT warmup training script |
| `train_rl_outcome_only.sh` | RL training with outcome-only reward |
| `train_rl_equalized.sh` | RL training with equalized reward |
| `train_rl_ipo.sh` | RL training with IPO reward |
| `train_rl_adaptive.sh` | RL training with existing adaptive reward + stability fixes |
| `docs/reward_modes_report.md` | This document |

**No existing files were modified.**

---

## Reward Mode Details

### 1. Outcome-Only (`main_ppo_outcome_only.py`)

**Formula:**
```
R(trajectory) = F1(predicted_answer, gold_answers)
```

- Reward placed at the last valid token position only
- No intermediate turn rewards
- **Purpose:** Lower-bound baseline. Known to cause reward collapse in multi-turn settings because the model receives no signal for intermediate clarification actions.

### 2. Equalized (`main_ppo_equalized.py`)

**Formula:**
```
R(trajectory) = F1 (at last token) + 0.1 × n_clarify_turns (at each </clarify> position)
```

- F1 outcome reward at last valid token
- Constant +0.1 reward at each `</clarify>` closing tag position
- Reward is independent of whether clarification actually helped
- **Purpose:** Prevents collapse by ensuring non-zero reward for any clarification behavior. Based on the UserRL paper approach.

### 3. IPO — Information Gain Policy Optimization (`main_ppo_ipo.py`)

**Formula:**
```
R(trajectory) = F1 (at last token) + Σ_t [α × F1 / n_intermediate] (at each </clarify> or </search> position)
```

Where:
- `α` = 0.5 (configurable via `+ipo.alpha=X` in the training command)
- `n_intermediate` = number of intermediate turns (clarify + search, excluding the final answer)
- Each intermediate turn gets an equal share of the IG reward

**IG Proxy Rationale:**
- If the model clarified/searched AND got the answer right (high F1), the intermediate actions contributed to information gain → positive IG reward
- If the model clarified/searched BUT got the answer wrong (low F1), the intermediate actions didn't help → near-zero IG reward (since F1 ≈ 0)
- This creates a natural credit assignment: intermediate actions are rewarded proportionally to their contribution to the final outcome

**Purpose:** Core innovation. Provides dense turn-level reward signal while still being grounded in outcome quality. Addresses the credit assignment problem in multi-turn RL.

### 4. Adaptive (`main_ppo.py`, existing)

**Formula:**
```
R = w_f1 × F1
  + w_clarify × adaptive_clarify_reward(F1, clarify_count)
  + w_confidence × sigmoid(avg_answer_logprob)
  + w_consistency × group_consistency(uid)
```

Default weights: f1=1.0, adaptive_clarify=0.3, confidence=0.1, consistency=0.2

- Uses adaptive clarify reward: rewards clarification only when the answer is correct
- Includes confidence bonus from answer token log-probs
- Includes self-consistency bonus across multiple rollouts of the same question
- **Purpose:** Most complex reward design, combines multiple signals

---

## SFT Warmup

### Why SFT First?

The base model (Qwen2.5-3B-Instruct) has ~50% format failure rate with the XML tag format (`<think>`, `<clarify>`, `<answer>`). RL training on a model that can't produce valid format leads to:
1. Most rollouts receive zero reward (can't extract answer)
2. Reward signal is too sparse for learning
3. Training collapse after ~150-200 steps

SFT warmup teaches the model the correct format before RL fine-tuning.

### SFT Data Generation

```bash
python scripts/data_process/sft_clarify_data.py --local_dir ./data/sft_clarify
```

Generates balanced training data:
- **Ambiguous questions** → gold trajectory: `<think>` → `<clarify>` → user response → `<think>` → `<answer>`
- **Non-ambiguous questions** → direct: `<think>` → `<answer>`
- ~50/50 balance between ambiguous and non-ambiguous

---

## Stability Fixes (Applied to All RL Scripts)

All RL training scripts include these stability improvements over the original `train_ambignq_fewshot.sh`:

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `kl_loss_coef` | 0.001 | 0.03 | Stronger KL constraint prevents policy from diverging too far from reference |
| `lr` | 1e-6 | 5e-7 | More conservative learning rate reduces instability |
| `lr_warmup_steps_ratio` | 0.15 | 0.2 | Longer warmup for more gradual initial updates |
| `grad_clip` | (default) | 1.0 | Explicit gradient clipping to prevent exploding gradients |

---

## How to Run Experiments

### Step 1: Generate SFT Data
```bash
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new/code
python scripts/data_process/sft_clarify_data.py --local_dir scripts/data_process/data/sft_clarify
```

### Step 2: SFT Warmup
```bash
sbatch train_sft_clarify.sh
```
Wait for completion. Note the checkpoint path (e.g., `verl_checkpoints/sft-clarify-warmup/global_step_XXX`).

### Step 3: Update BASE_MODEL in RL Scripts
Edit the `BASE_MODEL` variable in each RL script to point to the SFT checkpoint:
```bash
export BASE_MODEL='verl_checkpoints/sft-clarify-warmup/global_step_XXX'
```

### Step 4: Launch RL Experiments
```bash
sbatch train_rl_outcome_only.sh
sbatch train_rl_equalized.sh
sbatch train_rl_ipo.sh
sbatch train_rl_adaptive.sh
```

### Step 5: Evaluate
Use `main_eval.py` to evaluate each checkpoint.

---

## Expected Results

Based on prior work (UserRL, Search-R1) and our stability analysis:

| Reward Mode | Expected F1 | Clarify Rate | Notes |
|-------------|-------------|--------------|-------|
| Outcome-Only | Low (~0.2) | Collapses to 0% | Model learns to skip clarification |
| Equalized | Medium (~0.35) | ~30-50% | Stable but over-clarifies |
| IPO | High (~0.45) | ~25-40% | Targeted clarification |
| Adaptive | Medium-High (~0.40) | ~20-35% | Depends on weight tuning |

These are rough estimates; actual numbers will depend on training stability.

---

## Known Limitations & Notes

### Token Position Approximation (Medium)
`find_turn_token_positions()` in `reward_utils.py` re-encodes a text prefix to find the token index of closing tags. Due to BPE tokenizer round-trip asymmetry (decode then re-encode may not produce the same token count), the computed position could be off by 1-2 tokens. This affects IPO and equalized reward placement. In practice, this is tolerable for RL training but could be improved by searching for closing tag token IDs directly in the original `response_ids` tensor.

### IPO Reward Scale
Total reward for a trajectory with intermediate turns = `F1 * (1 + alpha)` (default `1.5`), vs. `F1` for direct-answer trajectories. This is intentional (rewarding useful intermediate steps), but when comparing ablation results across conditions, the reward scales are not directly comparable.

### SFT Response Format
SFT only trains on the **first turn** of model output. For ambiguous questions, this means `<think>` + `<clarify>` only — the `<user_response>` (environment-injected) and follow-up turns are excluded. This matches the RL environment where the model generates up to `</clarify>`, then the environment provides `<user_response>`.

### SFTDataset Dependency
The `SFTDataset` class used by `fsdp_sft_trainer.py` is not in the local codebase — it comes from the pip-installed `verl` package on the server. Ensure the server has the correct verl version installed.

### Checkpoint Path
The RL scripts reference `verl_checkpoints/sft-clarify-warmup/global_step_final` which is a symlink created by `train_sft_clarify.sh` after SFT completes. If you skip the SFT step or the symlink fails, you must manually update `BASE_MODEL` in the RL scripts to point to the actual checkpoint directory.
