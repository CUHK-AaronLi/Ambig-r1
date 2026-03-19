#!/bin/bash
#SBATCH --job-name="xeval-mt"
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

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code
mkdir -p out

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"

# ======================================================================
# Multi-Turn Cross-Dataset Evaluation (val_only mode)
#
# Uses the training pipeline's val_only mode to run full multi-turn
# evaluation (with clarify simulator) on 2 checkpoints x 5 datasets.
#
# Checkpoints:
#   1. pacific-cont  = PACIFIC-only best (step 100)
#   2. mix-b         = Mixed training best (step 50)
#
# Datasets: pacific, abgcoqa, sharc, situatedqa, ambignq
# ======================================================================

CKPT_BASE=/mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code/verl_checkpoints_diag

# --- Checkpoint definitions ---
declare -A CHECKPOINTS
CHECKPOINTS[pacific-cont]="$CKPT_BASE/ipo-v8a-pacific-cont/actor/global_step_100"
CHECKPOINTS[mixb]="$CKPT_BASE/ipo-mix-b-pac-sharc/actor/global_step_50"

# --- Dataset val file paths ---
declare -A VAL_FILES
VAL_FILES[pacific]=scripts/data_process/data/pacific_fewshot/validation.parquet
VAL_FILES[abgcoqa]=scripts/data_process/data/abgcoqa/val.parquet
VAL_FILES[sharc]=scripts/data_process/data/sharc_fewshot/tinytest.parquet
VAL_FILES[situatedqa]=scripts/data_process/data/situatedqa_fewshot/tinytest.parquet
VAL_FILES[ambignq]=scripts/data_process/data/ambignq_fewshot/tinytest.parquet

# Dummy train file (required by pipeline but unused in val_only mode)
DUMMY_TRAIN=scripts/data_process/data/pacific_fewshot/train.parquet

# SFT baselines for reference
declare -A SFT_F1
SFT_F1[pacific]=0.251
SFT_F1[abgcoqa]=0.153
SFT_F1[sharc]=0.438
SFT_F1[situatedqa]=0.129
SFT_F1[ambignq]=0.062

# Track results for final summary
RESULTS_LOG="eval_results/xeval_multiturn_summary.txt"
mkdir -p eval_results
echo "Cross-Dataset Multi-Turn Evaluation Summary" > $RESULTS_LOG
echo "Started: $(date)" >> $RESULTS_LOG
echo "==========================================" >> $RESULTS_LOG

# Validate checkpoints exist
for ckpt_name in pacific-cont mixb; do
    ckpt_path=${CHECKPOINTS[$ckpt_name]}
    if [ ! -d "$ckpt_path" ]; then
        echo "ERROR: Checkpoint not found: $ckpt_path"
        echo "  Available checkpoints under $CKPT_BASE:"
        ls -d $CKPT_BASE/*/actor/global_step_* 2>/dev/null
        exit 1
    fi
    echo "Checkpoint OK: $ckpt_name -> $ckpt_path"
done

EVAL_COUNT=0
TOTAL_EVALS=10

run_val_only() {
    local ckpt_name=$1
    local dataset=$2
    local ckpt_path=${CHECKPOINTS[$ckpt_name]}
    local val_file=${VAL_FILES[$dataset]}
    local experiment_name="xeval-${ckpt_name}-on-${dataset}"

    EVAL_COUNT=$((EVAL_COUNT + 1))

    echo ""
    echo "######################################################################"
    echo "# Evaluation $EVAL_COUNT / $TOTAL_EVALS"
    echo "# Checkpoint: $ckpt_name"
    echo "# Dataset:    $dataset"
    echo "# Experiment: $experiment_name"
    echo "# Val file:   $val_file"
    echo "# SFT baseline F1: ${SFT_F1[$dataset]}"
    echo "# Started at: $(date)"
    echo "######################################################################"
    echo ""

    if [ ! -f "$val_file" ]; then
        echo "WARNING: Val file not found: $val_file — SKIPPING"
        echo "$experiment_name: SKIPPED (val file missing)" >> $RESULTS_LOG
        return 1
    fi

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_ipo \
        data.train_files=$DUMMY_TRAIN \
        data.val_files=$val_file \
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
        actor_rollout_ref.model.path=$ckpt_path \
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
        +trainer.val_only=true \
        +trainer.val_before_train=true \
        trainer.total_training_steps=0 \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=1 \
        trainer.project_name=Ambig-R1 \
        trainer.total_epochs=1 \
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
        +ipo.alpha=0.1 \
        +ipo.turn_cost=0.0 \
        +ipo.enable_ablation=false \
        +ipo.efficiency_bonus=0.0 \
        +ipo.baseline_reward=0.0 \
        +ipo.clarify_bonus=0.0 \
        trainer.experiment_name=$experiment_name \
        trainer.default_local_dir=/mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code/verl_checkpoints_diag/$experiment_name \
        2>&1 | tee eval_results/${experiment_name}.log

    local exit_code=$?

    echo ""
    echo ">>> $experiment_name finished at $(date) with exit code $exit_code"
    echo "$experiment_name: exit_code=$exit_code (SFT_F1=${SFT_F1[$dataset]})" >> $RESULTS_LOG
    echo ""

    return $exit_code
}

# ======================================================================
# Main evaluation loop: 2 checkpoints x 5 datasets = 10 evaluations
# ======================================================================

echo ""
echo "===== Starting Multi-Turn Cross-Dataset Evaluation ====="
echo "Total evaluations: $TOTAL_EVALS (2 checkpoints x 5 datasets)"
echo ""

for ckpt_name in pacific-cont mixb; do
    echo ""
    echo "============================================================"
    echo "  Checkpoint: $ckpt_name"
    echo "  Path: ${CHECKPOINTS[$ckpt_name]}"
    echo "============================================================"
    echo ""

    for dataset in pacific abgcoqa sharc situatedqa ambignq; do
        run_val_only "$ckpt_name" "$dataset"
    done
done

# ======================================================================
# Final summary
# ======================================================================

echo ""
echo "######################################################################"
echo "#                    FINAL SUMMARY                                    "
echo "######################################################################"
echo ""
echo "SFT baselines: pacific=0.251, abgcoqa=0.153, sharc=0.438, situatedqa=0.129, ambignq=0.062"
echo ""
echo "All evaluation logs saved to: eval_results/xeval-*.log"
echo "Summary: $RESULTS_LOG"
echo ""
cat $RESULTS_LOG
echo ""
echo "Finished: $(date)" >> $RESULTS_LOG
echo "Job finished at: $(date)"
