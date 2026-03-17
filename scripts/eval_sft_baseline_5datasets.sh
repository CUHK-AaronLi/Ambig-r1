#!/bin/bash
#SBATCH --job-name="sft-eval"
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

SFT_MODEL=/mnt/users_home/cpii.local/yli/Ambig-R1-new/code/verl_checkpoints/sft-clarify-warmup/global_step_330
EVAL_DIR=eval_results/sft_baseline
mkdir -p $EVAL_DIR

echo "===== SFT Baseline Evaluation on 5 Datasets ====="
echo "Model: $SFT_MODEL"
echo "Purpose: Table 1 'Before RL' baseline row"
echo ""

# Dataset paths
declare -A DATASETS
DATASETS[pacific]=scripts/data_process/data/pacific_fewshot/validation.parquet
DATASETS[abgcoqa]=scripts/data_process/data/abgcoqa/validation.parquet
DATASETS[sharc]=scripts/data_process/data/sharc_fewshot/validation.parquet
DATASETS[situatedqa]=scripts/data_process/data/situatedqa_fewshot/validation.parquet
DATASETS[ambignq]=scripts/data_process/data/ambignq_fewshot/validation.parquet

for dataset in pacific abgcoqa sharc situatedqa ambignq; do
    data_path=${DATASETS[$dataset]}
    output_path=$EVAL_DIR/sft_${dataset}.json

    echo "===== Evaluating on $dataset ====="
    echo "Data: $data_path"
    echo "Output: $output_path"

    if [ ! -f "$data_path" ]; then
        echo "WARNING: $data_path does not exist, skipping"
        continue
    fi

    python3 scripts/eval_checkpoint.py \
        --model_path $SFT_MODEL \
        --data_path $data_path \
        --output_path $output_path \
        --max_samples 100 \
        --max_new_tokens 2048

    echo "Done: $dataset"
    echo ""
done

echo ""
echo "===== All evaluations complete ====="
echo "Results in: $EVAL_DIR/"
ls -la $EVAL_DIR/

echo "Job finished at: $(date)"
