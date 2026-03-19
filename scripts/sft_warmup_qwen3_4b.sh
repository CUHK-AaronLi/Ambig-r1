#!/bin/bash
#SBATCH --job-name="sft-q3-4b"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=07-00:00:00
#SBATCH --gpus=4
#SBATCH --exclude=CPIIGPU-211-128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1155124348@link.cuhk.edu.hk

hostname
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "===== SFT Warmup: Qwen3-4B BASE (3 epochs, save per epoch) ====="
echo "NOTE: Qwen3-4B is a BASE model (not Instruct), may need more epochs to learn format."

mkdir -p out

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1

cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code

export WANDB_PROJECT='Ambig-R1'
export BASE_MODEL='Qwen/Qwen3-4B'
export EXPERIMENT_NAME=sft-warmup-qwen3-4b
export WANDB_MODE=offline

# SFT data (in user's directory)
SFT_DATA_DIR=/mnt/users_home/cpii.local/yli/Ambig-R1-new/code/scripts/data_process/data/sft_clarify

# Verify data exists
if [ ! -f "$SFT_DATA_DIR/train.parquet" ]; then
    echo "ERROR: SFT data not found at $SFT_DATA_DIR/train.parquet"
    SFT_DATA_DIR=scripts/data_process/data/sft_clarify
    if [ ! -f "$SFT_DATA_DIR/train.parquet" ]; then
        echo "ERROR: SFT data not found in either location. Aborting."
        exit 1
    fi
fi
echo "Using SFT data from: $SFT_DATA_DIR"

# Verify model is accessible
echo "Base model: $BASE_MODEL"
python3 -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('$BASE_MODEL'); print(f'Model: {c.architectures}, hidden={c.hidden_size}, layers={c.num_hidden_layers}')" 2>&1 || {
    echo "WARNING: Cannot access $BASE_MODEL. May need to pre-download."
}

# ---------- SFT Training: 3 epochs → checkpoints at ~330, ~660, ~990 ----------
# Base model typically needs more training than Instruct models.
# Expect val_loss to keep decreasing for 2-3 epochs.
echo ""
echo "Training 3 epochs. SFT trainer saves checkpoint at end of each epoch."
echo "Check val/loss in the log to pick the best epoch."
echo ""

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
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=3 \
    trainer.total_training_steps=null \
    trainer.validate_before_training=True \
    trainer.logger='[console,wandb]' \
    trainer.save_freq=110 \
    2>&1 | tee $EXPERIMENT_NAME.log

# ---------- Summary ----------
echo ""
echo "===== Val Loss Summary ====="
grep -E "val/loss" $EXPERIMENT_NAME.log | tail -5
echo ""
echo "Checkpoints saved in: verl_checkpoints/$EXPERIMENT_NAME/"
ls -la verl_checkpoints/$EXPERIMENT_NAME/global_step_* 2>/dev/null
echo ""
echo "NEXT STEP: Pick the checkpoint with lowest val_loss for RL training."

CKPT_DIR="verl_checkpoints/$EXPERIMENT_NAME"
LATEST_CKPT=$(ls -td ${CKPT_DIR}/global_step_* 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    ln -sfn "$(basename $LATEST_CKPT)" "${CKPT_DIR}/global_step_final"
    echo "Created symlink: ${CKPT_DIR}/global_step_final -> $(basename $LATEST_CKPT)"
fi

echo "Job finished at: $(date)"
