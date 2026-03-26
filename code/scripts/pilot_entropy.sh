#!/bin/bash
#SBATCH --job-name="entropy"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=2:00:00
#SBATCH --gpus=4
#SBATCH --exclude=CPIIGPU-211-128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

hostname
echo "Job started at: $(date)"

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 scripts/pilot_entropy_analysis.py \
    --model_path verl_checkpoints/sft-7b-pacific/global_step_best \
    --val_data scripts/data_process/data/pacific_fewshot/validation.parquet \
    --n_samples 100 \
    --n_completions 5 \
    --max_tokens 128 \
    --temperature 1.0

echo "Job finished at: $(date)"
