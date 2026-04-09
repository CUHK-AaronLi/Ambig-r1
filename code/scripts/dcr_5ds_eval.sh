#!/bin/bash
#SBATCH --job-name="dcr-5ds-eval"
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-%x.out
#SBATCH -e out/%j-%x.err
#SBATCH --time=12:00:00
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

hostname
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)"
conda activate searchr1
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code
mkdir -p out

# DCR checkpoint (job 8631, 100 steps)
export BASE_MODEL=/mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code/verl_checkpoints_diag/dcr-diag-100/actor/global_step_100
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"

# Common args — DCR mode ON
COMMON_ARGS="
actor_rollout_ref.model.path=$BASE_MODEL
actor_rollout_ref.model.enable_gradient_checkpointing=true
actor_rollout_ref.model.use_remove_padding=False
actor_rollout_ref.actor.optim.lr=5e-7
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size=8
actor_rollout_ref.actor.grad_clip=1.0
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.grad_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
actor_rollout_ref.rollout.log_prob_micro_batch_size=16
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.ref.log_prob_micro_batch_size=16
actor_rollout_ref.ref.fsdp_config.param_offload=False
actor_rollout_ref.actor.kl_loss_coef=0.04
actor_rollout_ref.actor.kl_loss_type=low_var_kl
algorithm.no_think_rl=false
actor_rollout_ref.rollout.n_agent=5
actor_rollout_ref.rollout.temperature=1
actor_rollout_ref.actor.state_masking=true
algorithm.adv_estimator=grpo
+ambigqa.enable_search_action=false
trainer.logger=['console']
+trainer.val_only=true
trainer.n_gpus_per_node=4
trainer.nnodes=1
trainer.save_freq=999
trainer.test_freq=1
trainer.project_name=Ambig-R1
trainer.total_epochs=1
trainer.total_training_steps=2
trainer.default_hdfs_dir=null
trainer.num_cpus=20
max_turns=4
+ambigqa.enable_clarify_action=true
+ambigqa.max_clarify_turns=3
+ambigqa.gpt4_simulator_url=\"http://10.10.211.118:8001/batch_generate\"
+ambigqa.azure_openai_endpoint=\"https://cpii-s5.openai.azure.com/\"
+ambigqa.azure_openai_api_key=\"91e5ea9bf61c4769a44b0b0b5c67d559\"
+ambigqa.azure_openai_deployment=\"gpt-4o\"
+ambigqa.enable_entropy=false
+ipo.alpha=0.3
+ipo.turn_cost=0.0
+ipo.enable_ablation=true
+ipo.counterfactual_logprob=false
+ipo.efficiency_bonus=0.0
+ipo.baseline_reward=0.0
+ipo.clarify_bonus=0.15
+ipo.ig_threshold=0.0
+ipo.ambiguity_penalty=0.15
+ipo.outcome_scale=1.0
+ipo.dcr_mode=true
data.train_batch_size=32
data.val_batch_size=32
data.max_prompt_length=8192
data.max_response_length=2048
data.max_start_length=3072
data.max_obs_length=512
data.shuffle_train_dataloader=False
"

# All 5 datasets
DATASETS="pacific_fewshot ambignq_fewshot abgcoqa sharc_fewshot situatedqa_fewshot"

RESULTS_FILE="dcr_5ds_results.txt"
echo "# DCR Checkpoint Cross-Domain Evaluation" > $RESULTS_FILE
echo "# $(date)" >> $RESULTS_FILE
echo "# Model: $BASE_MODEL" >> $RESULTS_FILE
echo "# DCR mode: ON" >> $RESULTS_FILE
echo "#" >> $RESULTS_FILE
echo "# Dataset | F1 | ClarifyRate | post_clarify_F1 | no_clarify_F1 | avg_turns" >> $RESULTS_FILE

for DS in $DATASETS; do
    DS_DIR=scripts/data_process/data/$DS
    EXPNAME=dcr-5ds-${DS}

    echo ""
    echo "===== Evaluating DCR checkpoint on $DS ====="
    echo "Data dir: $DS_DIR"

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_ipo \
        data.train_files=$DS_DIR/train.parquet \
        data.val_files=$DS_DIR/validation.parquet \
        data.train_data_num=16 \
        data.val_data_num=100 \
        $COMMON_ARGS \
        trainer.experiment_name=$EXPNAME \
        trainer.default_local_dir=/tmp/$EXPNAME \
        2>&1 | tee ${EXPNAME}.log

    echo "===== $DS Results ====="
    LOG="${EXPNAME}.log"
    F1=$(grep "val/f1/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/f1/\w+:\K[0-9.]+' | head -1)
    CLAR=$(grep "val/clarify_rate/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/clarify_rate/\w+:\K[0-9.]+' | head -1)
    PCLAR=$(grep "val/post_clarify_f1/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/post_clarify_f1/\w+:\K[0-9.]+' | head -1)
    NCLAR=$(grep "val/no_clarify_f1/" $LOG 2>/dev/null | tail -1 | grep -oP 'val/no_clarify_f1/\w+:\K[0-9.]+' | head -1)
    AVGTRN=$(grep "env/number_of_actions/mean:" $LOG 2>/dev/null | tail -1 | grep -oP 'env/number_of_actions/mean:\K[0-9.]+' | head -1)

    echo "$DS | $F1 | $CLAR | $PCLAR | $NCLAR | $AVGTRN" >> $RESULTS_FILE
    echo "  → F1=$F1, clarify=$CLAR, post_clarify_f1=$PCLAR, no_clarify_f1=$NCLAR"
    echo ""
done

echo ""
echo "===== DCR Cross-Domain Results ====="
cat $RESULTS_FILE
echo ""
echo "===== Comparison: DCR vs Baseline (ar-best-200) ====="
echo ""
echo "Dataset       | Baseline F1 | DCR F1  | ΔF1 | Baseline clarify | DCR clarify | Δclarify"
echo "------------- | ----------- | ------- | --- | ---------------- | ---------- | ---------"
echo "pacific_fewshot | 0.680     | [TBD]   | [TBD] | 0.33 | [TBD] | [TBD]"
echo "ambignq_fewshot | 0.159     | [TBD]   | [TBD] | 0.15 | [TBD] | [TBD]"
echo "abgcoqa         | 0.309     | [TBD]   | [TBD] | 0.29 | [TBD] | [TBD]"
echo "sharc_fewshot   | [TBD]     | [TBD]   | [TBD] | [TBD] | [TBD] | [TBD]"
echo "situatedqa_fewshot | [TBD]  | [TBD]   | [TBD] | [TBD] | [TBD] | [TBD]"
echo ""
echo "Done at: $(date)"
