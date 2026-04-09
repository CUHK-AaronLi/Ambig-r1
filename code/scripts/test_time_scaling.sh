#!/bin/bash
#SBATCH --job-name="tts-eval"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=6:00:00
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
# Test-Time Scaling Analysis: F1 vs max_turns
# Evaluates ar-best-200 checkpoint at different max_turns settings

hostname
echo "===== Test-Time Scaling Analysis ====="
echo "Started at: $(date)"

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code
mkdir -p out

export BASE_MODEL=/mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code/verl_checkpoints_diag/ar-best-200/actor/global_step_180
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"

# Models to test
MODEL_LIST="ambignq_fewshot abgcoqa"

echo ""
echo "===== TURN SETTINGS ====="
echo "max_turns: 0, 1, 2, 3"
echo "Datasets: ambignq_fewshot, abgcoqa"
echo "Checkpoint: global_step_180 (ar-best-200)"
echo ""

RESULTS_FILE="tts_results.txt"
echo "# Test-Time Scaling Results" > $RESULTS_FILE
echo "# $(date)" >> $RESULTS_FILE
echo "# max_turns, dataset, f1, clarify_rate, post_clarify_f1, no_clarify_f1, avg_turns" >> $RESULTS_FILE

for DS in $MODEL_LIST; do
    for TURNS in 0 1 2 3; do
        EXPNAME="tts-${DS}-t${TURNS}"
        echo ""
        echo "===== Dataset=$DS, max_turns=$TURNS ====="

        # Use validation.parquet if exists, else test.parquet
        VAL_FILE="scripts/data_process/data/${DS}/validation.parquet"
        if [ ! -f "$VAL_FILE" ]; then
            VAL_FILE="scripts/data_process/data/${DS}/test.parquet"
        fi

        PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_ipo \
            data.train_files=scripts/data_process/data/${DS}/train.parquet \
            data.val_files=$VAL_FILE \
            data.train_data_num=64 \
            data.val_data_num=100 \
            data.train_batch_size=32 \
            data.val_batch_size=32 \
            data.max_prompt_length=8192 \
            data.max_response_length=2048 \
            data.max_start_length=3072 \
            data.max_obs_length=512 \
            data.shuffle_train_dataloader=False \
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
            +ambigqa.enable_search_action=false \
            trainer.logger=['console'] \
            +trainer.val_only=true \
            +trainer.val_before_train=false \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.save_freq=999 \
            trainer.test_freq=1 \
            trainer.project_name=Ambig-R1 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=2 \
            trainer.default_hdfs_dir=null \
            trainer.num_cpus=20 \
            max_turns=$TURNS \
            +ambigqa.enable_clarify_action=true \
            +ambigqa.max_clarify_turns=$TURNS \
            +ambigqa.gpt4_simulator_url="http://10.10.211.118:8001/batch_generate" \
            +ambigqa.azure_openai_endpoint="https://cpii-s5.openai.azure.com/" \
            +ambigqa.azure_openai_api_key="91e5ea9bf61c4769a44b0b0b5c67d559" \
            +ambigqa.azure_openai_deployment="gpt-4o" \
            +ambigqa.enable_entropy=false \
            +ipo.alpha=0.0 \
            +ipo.clarify_bonus=0.0 \
            +ipo.ambiguity_penalty=0.0 \
            +ipo.outcome_scale=1.0 \
            trainer.experiment_name=$EXPNAME \
            trainer.default_local_dir=/tmp/$EXPNAME \
            2>&1 | tee ${EXPNAME}.log &

        # Wait for this to finish before starting next
        wait

        # Extract metrics
        LOG="${EXPNAME}.log"
        F1=$(grep "val/f1/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/f1/\w+:\K[0-9.]+' | head -1)
        CLAR=$(grep "val/clarify_rate/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/clarify_rate/\w+:\K[0-9.]+' | head -1)
        PCLAR=$(grep "val/post_clarify_f1/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/post_clarify_f1/\w+:\K[0-9.]+' | head -1)
        NCLAR=$(grep "val/no_clarify_f1/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/no_clarify_f1/\w+:\K[0-9.]+' | head -1)
        AVGTRN=$(grep "env/number_of_actions/mean:" $LOG 2>/dev/null | tail -1 | grep -oP 'env/number_of_actions/mean:\K[0-9.]+' | head -1)

        echo "$TURNS, $DS, $F1, $CLAR, $PCLAR, $NCLAR, $AVGTRN" >> $RESULTS_FILE
        echo "  → F1=$F1, clarify_rate=$CLAR, post_clarify_f1=$PCLAR, no_clarify_f1=$NCLAR"
    done
done

echo ""
echo "===== Final Results ====="
cat $RESULTS_FILE
echo ""
echo "Done at: $(date)"
