#!/bin/bash
#SBATCH --job-name="p0-xeval"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=04:00:00
#SBATCH --gpus=2
#SBATCH --exclude=CPIIGPU-211-128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1155124348@link.cuhk.edu.hk

hostname
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code

# ======================================================================
# Phase 0: Cross-Dataset Transfer Evaluation
#
# Purpose: Evaluate PACIFIC-trained checkpoint on ALL 5 val sets to see
# if a single-dataset model generalizes across ambiguity types.
#
# Decision point:
#   If F1 > SFT baseline on non-PACIFIC datasets → PACIFIC generalizes
#   If F1 ≈ SFT or worse → need mixed training (Phase 1)
# ======================================================================

# PACIFIC best checkpoint — try best available (cont > cont2 > original)
CKPT_BASE=/mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code/verl_checkpoints_diag
CKPT=""
for candidate in \
    "$CKPT_BASE/ipo-v8a-pacific-cont/actor/global_step_100" \
    "$CKPT_BASE/ipo-v8a-pacific-cont2/actor/global_step_50" \
    "$CKPT_BASE/ipo-v8a-pacific/actor/global_step_100"; do
    if [ -d "$candidate" ]; then
        CKPT=$candidate
        break
    fi
done

if [ -z "$CKPT" ]; then
    echo "ERROR: No PACIFIC checkpoint found. Available:"
    ls -d $CKPT_BASE/ipo-v8a*/actor/global_step_* 2>/dev/null
    exit 1
fi

SFT_MODEL=/mnt/users_home/cpii.local/yli/Ambig-R1-new/code/verl_checkpoints/sft-clarify-warmup/global_step_330

EVAL_DIR=eval_results/phase0_cross_dataset
mkdir -p $EVAL_DIR

echo "===== Phase 0: Cross-Dataset Transfer Evaluation ====="
echo "PACIFIC checkpoint: $CKPT"
echo ""

# Dataset paths (same as v8 training val files)
declare -A DATASETS
DATASETS[pacific]=scripts/data_process/data/pacific_fewshot/validation.parquet
DATASETS[abgcoqa]=scripts/data_process/data/abgcoqa/val.parquet
DATASETS[sharc]=scripts/data_process/data/sharc_fewshot/tinytest.parquet
DATASETS[situatedqa]=scripts/data_process/data/situatedqa_fewshot/tinytest.parquet
DATASETS[ambignq]=scripts/data_process/data/ambignq_fewshot/tinytest.parquet

# SFT baselines for comparison (from P1-A1)
declare -A SFT_F1
SFT_F1[pacific]=0.251
SFT_F1[abgcoqa]=0.153
SFT_F1[sharc]=0.438
SFT_F1[situatedqa]=0.129
SFT_F1[ambignq]=0.062

echo "===== Evaluating PACIFIC checkpoint on 5 datasets ====="
echo ""

for dataset in pacific abgcoqa sharc situatedqa ambignq; do
    data_path=${DATASETS[$dataset]}
    output_path=$EVAL_DIR/pacific_on_${dataset}.json
    sft_baseline=${SFT_F1[$dataset]}

    echo "===== $dataset (SFT baseline F1=$sft_baseline) ====="
    echo "Data: $data_path"

    if [ ! -f "$data_path" ]; then
        echo "WARNING: $data_path does not exist, skipping"
        continue
    fi

    python3 scripts/eval_checkpoint.py \
        --model_path $CKPT \
        --data_path $data_path \
        --output_path $output_path \
        --max_samples 100 \
        --max_new_tokens 2048

    echo "Done: $dataset"
    echo ""
done

echo ""
echo "===== Summary: Cross-Dataset Transfer ====="
echo "SFT baselines: pacific=0.251, abgcoqa=0.153, sharc=0.438, situatedqa=0.129, ambignq=0.062"
echo ""
echo "PACIFIC checkpoint results:"
for dataset in pacific abgcoqa sharc situatedqa ambignq; do
    output_path=$EVAL_DIR/pacific_on_${dataset}.json
    if [ -f "$output_path" ]; then
        f1=$(python3 -c "import json; d=json.load(open('$output_path')); print(f\"{d['summary']['f1_mean']:.4f}\")" 2>/dev/null || echo "N/A")
        clarify=$(python3 -c "import json; d=json.load(open('$output_path')); print(f\"{d['summary']['avg_clarify_per_sample']:.2f}\")" 2>/dev/null || echo "N/A")
        echo "  $dataset: F1=$f1, avg_clarify=$clarify (SFT=${SFT_F1[$dataset]})"
    fi
done

echo ""
echo "Decision: If non-PACIFIC F1 > SFT baselines above → PACIFIC generalizes"
echo "           If F1 ≈ SFT or worse → need mixed training (Phase 1)"

echo ""
echo "All evaluations saved to: $EVAL_DIR/"
ls -la $EVAL_DIR/

echo "Job finished at: $(date)"
