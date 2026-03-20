#!/bin/bash
#SBATCH --job-name="p4-anq-arew"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=12:00:00
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
echo "===== P4: AmbigNQ + AReW Critique (100 steps) ====="
echo "NOTE: p3 showed classic self-locking (tool collapse by step 90)."
echo "      AReW adds step-level critique to break self-locking cycle."
echo "      Formula: Â_t = A_t + λ_as * u_as + λ_multi * u_multi"

mkdir -p out

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code
mkdir -p out

export DATA_DIR=scripts/data_process/data/ambignq_fewshot
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export BASE_MODEL=/mnt/users_home/cpii.local/yli/Ambig-R1-new/code/verl_checkpoints/sft-clarify-warmup/global_step_330
export EXPERIMENT_NAME=p4-ambignq-arew

# Verify SFT checkpoint
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: SFT checkpoint not found at $BASE_MODEL"
    exit 1
fi

# AmbigNQ: search+clarify ENABLED + AReW critique
# Compared to p3 (no AReW): expect tools to survive past step 90
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_ipo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/tinytest.parquet \
    data.train_data_num=null \
    data.val_data_num=100 \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
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
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    retriever.url="http://10.10.211.118:8000/retrieve" \
    retriever.topk=3 \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=10 \
    trainer.project_name=Ambig-R1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.default_hdfs_dir=null \
    trainer.num_cpus=40 \
    max_turns=4 \
    +ambigqa.enable_clarify_action=true \
    +ambigqa.max_clarify_turns=3 \
    +ambigqa.gpt4_simulator_url="http://10.10.211.118:8001/batch_generate" \
    +ambigqa.azure_openai_endpoint="https://cpii-s5.openai.azure.com/" \
    +ambigqa.azure_openai_api_key="91e5ea9bf61c4769a44b0b0b5c67d559" \
    +ambigqa.azure_openai_deployment="gpt-4o" \
    +ambigqa.enable_entropy=false \
    +ambigqa.enable_search_action=true \
    +ipo.alpha=0.1 \
    +ipo.turn_cost=0.0 \
    +ipo.enable_ablation=false \
    +ipo.efficiency_bonus=0.0 \
    +ipo.baseline_reward=0.0 \
    +ipo.clarify_bonus=0.0 \
    +arew.enable=true \
    +arew.lambda_as=0.3 \
    +arew.lambda_multi=0.05 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=/mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code/verl_checkpoints_diag/$EXPERIMENT_NAME \
    2>&1 | tee $EXPERIMENT_NAME.log

echo ""
echo "===== AmbigNQ + AReW Results ====="
echo "Compare with:"
echo "  p3 (no AReW): F1=0.155, tool collapse at step 90"
echo "  Expected: tools survive past step 90, F1 > 0.155"
grep -E "val/test_score|val/f1|val/clarify|val/search|arew/" $EXPERIMENT_NAME.log | tail -30
echo ""
echo "Job finished at: $(date)"
