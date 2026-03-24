#!/bin/bash

## SFT Warmup 7B with PACIFIC-enriched data
#SBATCH --job-name="sft-7b-pac"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=07-00:00:00
#SBATCH --gpus=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1155124348@link.cuhk.edu.hk

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

hostname
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "===== SFT 7B Warmup: Clarify + PACIFIC Data ====="

mkdir -p out

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1

cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code

export WAND_PROJECT='Ambig-R1'
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME=sft-7b-pacific
export WANDB_MODE=offline

# ---------- Step 1: Generate combined SFT data (if not already done) ----------
SFT_DATA_DIR='scripts/data_process/data/sft_pacific_7b'
if [ ! -f "$SFT_DATA_DIR/train.parquet" ]; then
    echo "Generating combined SFT + PACIFIC training data..."
    python3 scripts/prepare_sft_pacific.py \
        --sft_data scripts/data_process/data/sft_clarify/train.parquet \
        --pacific_data scripts/data_process/data/pacific_fewshot/train.parquet \
        --output_dir $SFT_DATA_DIR \
        --n_pacific 3000 \
        --seed 42
    echo "SFT data generation complete."
else
    echo "SFT data already exists at $SFT_DATA_DIR"
fi

# ---------- Step 2: SFT Training ----------
echo "Starting SFT training on Qwen2.5-7B-Instruct with PACIFIC-enriched data..."
PYTHONUNBUFFERED=1 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$SFT_DATA_DIR/train.parquet \
    data.val_files=$SFT_DATA_DIR/val.parquet \
    data.train_batch_size=32 \
    data.micro_batch_size=4 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=4096 \
    data.truncation=error \
    data.balance_dp_token=False \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.wrap_policy.min_num_params=0 \
    model.fsdp_config.cpu_offload=False \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=False \
    optim.lr=1e-5 \
    optim.betas='[0.9,0.95]' \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=null \
    trainer.validate_before_training=True \
    trainer.logger='[console,wandb]' \
    2>&1 | tee $EXPERIMENT_NAME.log

# ---------- Step 3: Create symlinks for RL scripts to reference ----------
CKPT_DIR="verl_checkpoints/$EXPERIMENT_NAME"
LATEST_CKPT=$(ls -td ${CKPT_DIR}/global_step_* 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    ln -sfn "$(basename $LATEST_CKPT)" "${CKPT_DIR}/global_step_final"
    echo "Created symlink: ${CKPT_DIR}/global_step_final -> $(basename $LATEST_CKPT)"

    # Also create global_step_best pointing to final (can be updated manually)
    ln -sfn "$(basename $LATEST_CKPT)" "${CKPT_DIR}/global_step_best"
    echo "Created symlink: ${CKPT_DIR}/global_step_best -> $(basename $LATEST_CKPT)"
else
    echo "WARNING: No checkpoint found in ${CKPT_DIR}"
fi

echo "Job finished at: $(date)"
