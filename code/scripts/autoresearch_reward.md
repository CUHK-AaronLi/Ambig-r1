# AutoResearch: Reward Design Optimization

## Goal
Find a reward function that achieves **selective clarification**:
- F1 > 0.50
- clarify_rate_ambig > 0.30 (clarify ambiguous questions)
- clarify_rate_non_ambig < clarify_rate_ambig (selective, not uniform)

## Fixed (DO NOT CHANGE)
- Training framework: verl + GRPO
- Model: Qwen3-4B, SFT checkpoint: sft-clarify-warmup/global_step_330
- Data: PACIFIC (pacific_fewshot), 12.4% req_clari=True
- Evaluation: 30-step pilot, 4 GPU, test_freq=10
- Base script template: abl_3b_ambig_gated.sh

## Mutable (WHAT TO OPTIMIZE)
The reward computation in `main_ppo_ipo.py` `IPORewardManager.__call__()`.

Current reward structure:
```
For each sample:
  1. F1 outcome reward at last token
  2. Per-turn IG reward (ablation-based) at turn tokens
  3. Turn cost penalty
  4. Ambiguity-aware penalty
  5. Clarify bonus
```

Parameters to tune:
- alpha (IG weight): 0.1-1.0
- turn_cost: 0.0-0.1
- clarify_bonus: 0.0-0.3
- ambiguity_penalty: 0.0-0.5
- ig_threshold: 0.0-0.3
- Whether to gate IG off for non-ambig samples
- Whether to flip IG sign for non-ambig
- Outcome reward scaling (e.g., 0.5 * F1 instead of F1)

## Evaluation Metric
```python
score = 0.0
if F1 < 0.40:
    score = F1  # Too low, just optimize F1
elif clarify_rate == 0:
    score = F1 * 0.5  # Penalize no clarification
else:
    selectivity = clarify_rate_ambig - clarify_rate_non_ambig
    score = F1 * 0.4 + min(clarify_rate, 0.5) * 0.3 + max(selectivity, 0) * 0.3
```

## Experiment History
| Run | Config | F1 | clr | clr_ambig | clr_nonambig | Score | Keep? |
|-----|--------|-----|-----|-----------|-------------|-------|-------|
| baseline (v8a) | alpha=0.1 | 0.599 | 50% | ? | ? | ~0.50 | baseline |
| igt=0.20 | +threshold=0.20 | 0.603 | 46% | ? | ? | ~0.51 | ✓ |
| ambig_penalty=0.3+gating | +penalty+gate_ig | 0.506 | 0% | - | - | 0.25 | ✗ |

## Ideas to Try
1. Outcome scaling: F1 * 0.5 (reduce outcome dominance)
2. Large clarify_bonus=0.2 + ambig_penalty=0.2 (carrot + stick)
3. IG sign flip for non-ambig (negative IG discourages non-ambig clarify)
4. Separate alpha: alpha_ambig=0.3, alpha_nonambig=0.05
5. Reward = IG only, no F1 outcome (radical: let IG be the only signal)
6. Two-phase: first 50 steps force clarify (bonus=0.3), then remove bonus
