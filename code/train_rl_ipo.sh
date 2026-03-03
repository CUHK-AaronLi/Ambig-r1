#!/bin/bash

## RL Training: IPO Reward (Information Gain Policy Optimization)
#SBATCH --job-name="rl-ipo"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=07-00:00:00
#SBATCH --gpus=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1155124348@link.cuhk.edu.hk
#SBATCH --nodelist=CPIIGPU-211-135

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

hostname
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "===== RL: IPO Reward (F1 + IG turn reward) ====="

mkdir -p out

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1

cd /mnt/users_home/cpii.local/yli/Ambig-R1-new/code

export DATA_DIR='scripts/data_process/data/ambignq_fewshot'
export WAND_PROJECT='Ambig-R1'
# Use SFT warmup checkpoint as base model (change path after SFT completes)
export BASE_MODEL='verl_checkpoints/sft-clarify-warmup/global_step_final'
export EXPERIMENT_NAME=rl-ipo
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline

# Azure OpenAI (GPT simulator for clarify)
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"

# ---------- Retrieval Server ----------
RETRIEVAL_URL="http://10.10.211.118:8000/retrieve"
echo "Using Retrieval Server at ${RETRIEVAL_URL}"

# ---------- GPT Simulator ----------
SIMULATOR_URL="http://10.10.211.118:8001"
echo "Using GPT4 Simulator at ${SIMULATOR_URL}"

# ---------- Training (stability fixes applied) ----------
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_ipo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/tinytest.parquet \
    data.train_data_num=null \
    data.val_data_num=200 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.max_start_length=3072 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.kl_loss_coef=0.03 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    retriever.url="${RETRIEVAL_URL}" \
    retriever.topk=3 \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.num_cpus=40 \
    max_turns=4 \
    +ambigqa.enable_clarify_action=true \
    +ambigqa.max_clarify_turns=3 \
    +ambigqa.gpt4_simulator_url="http://10.10.211.118:8001/batch_generate" \
    +ambigqa.azure_openai_endpoint="https://cpii-s5.openai.azure.com/" \
    +ambigqa.azure_openai_api_key="91e5ea9bf61c4769a44b0b0b5c67d559" \
    +ambigqa.azure_openai_deployment="gpt-4o" \
    +ambigqa.enable_entropy=false \
    +ipo.alpha=0.5 \
    2>&1 | tee $EXPERIMENT_NAME.log

echo "Job finished at: $(date)"
