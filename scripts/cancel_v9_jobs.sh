#!/bin/bash
# Cancel v9 search-fix experiments + v8j (depends on v9)
# These are being replaced by the systematic one-variable-at-a-time approach.
#
# Cancelled:
#   7360 - v9e AmbigNQ (search fix)
#   7361 - v9d SituatedQA (search fix + tc)
#   7362 - v9a PACIFIC (search fix + 400 steps)
#   7363 - v9c ShARC (search fix + tc)
#   7356 - v8j outcome-only SituatedQA (depends on v9)
#
# Kept:
#   7355 - v8i (alpha=0.2 PACIFIC, running) → α ablation data
#   7326 - tc01 (PACIFIC turn_cost=0.01, running) → tc ablation data
#   7364 - sft-eval (SFT baseline) → needed for Table 1
#   7367 - structnav Phase B → separate paper
#   7368-7370 - tc ablation (PACIFIC tc=0.02, SitQA tc=0.005, ShARC tc=0.01)

echo "Cancelling v9 search-fix experiments + v8j..."
scancel 7360 7361 7362 7363 7356

echo "Cancelled jobs: 7360 7361 7362 7363 7356"
echo ""
echo "Verifying..."
squeue -u yli
