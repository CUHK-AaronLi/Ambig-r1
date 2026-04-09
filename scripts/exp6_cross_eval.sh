#!/bin/bash
#SBATCH --job-name="exp6-eval"
 # Cross-dataset zero-shot evaluation of ar-best-200 checkpoint on AmbigNQ/AbgCoQA
#SBATCH --account=pgs
#SBATCH --qos=low
#SBATCH --partition=gemini
#SBATCH -o out/%j-150x.out
#SBATCH -e out/%j-150x.err
#SBATCH --time=24:00:00
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

hostname
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
 echo "===== EXP6: Zero-shot cross-dataset eval of ar-best-200 (PACIFIC) =====" echo "Model: ar-best-200/actor/global_step_180 (F1=0.680 peak at step 190)"

 echo "Evaluates on AmbigNQ and AbgCoQA ( PACIFIC-only training, zero-shot"

 source ~/anaconda3/bin/activate
eval "$(conda shell.bash hook)" || true && conda activate searchr1
 cd /mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code

 mkdir -p out

 export MODEL_PATH=/mnt/users_home/cpii.local/yli/Ambig-R1-new-claude/code/verl_checkpoints_diag/ar-best-200/actor/global_step_180
 export VLLM_ATTENTION_BACKEND=XFORMERS
 export WANDB_MODE=offline
 export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/" export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559" export AZURE_DEPLOYMENT="gpt-4o" export AZURE_API_VERSION="2024-02-01"

 if [ ! -d "$MODEL_PATH" ]; then
 echo "ERROR: Checkpoint not found at $MODEL_PATH" exit 1
 fi

echo "===== Evaluating on AmbigNQ ====="
python3 -c verl.trainer.main_ppo_ipo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    data.train_files=scripts/data_process/data/ambignq_fewshot/train.parquet \
    data.val_files=scripts/data_process/data/ambignq_fewshot/test.parquet \
    data.train_data_num=100 \
    data.val_data_num=100 \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.max_start_length=3072 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=0 \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=0 \
    trainer.default_hdfs_dir=null \
    max_turns=4 \
    +ambigqa.enable_clarify_action=true \
    +ambigqa.max_clarify_turns=3 \
    +ambigqa.gpt4_simulator_url="http://10.10.211.118:8001/batch_generate" \
    +ambigqa.azure_openai_endpoint="https://cpii-s5.openai.azure.com/" \
    +ambigqa.azure_openai_api_key="91e5ea9bf61c4769a44b0b0b5c67d559" \
    +ambigqa.azure_openai_deployment="gpt-4o" \
    +ambigqa.azure_openai_api_version="2024-02-01" \
    +ambigqa.enable_entropy=false \
    +ipo.alpha=0.3 \
    +ipo.turn_cost=0.0 \
    +ipo.enable_ablation=true \
    +ipo.counterfactual_logprob=false \
    +ipo.efficiency_bonus=0.0 \
    +ipo.baseline_reward=0.0 \
    +ipo.clarify_bonus=0.15 \
    +ipo.ig_threshold=0.0 \
    +ipo.ambiguity_penalty=0.15 \
    +ipo.outcome_scale=1.0 \
    trainer.experiment_name=eval6-cross-eval-ambignq \
    2>&1 | tee exp6_cross_eval_ambignq.log

echo ""
echo "===== Now evaluating on AbgCoQA ====="
python3 -c verl.trainer.main_ppo_ipo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    data.train_files=scripts/data_process/data/abgcoqa/train.parquet \
    data.val_files=scripts/data_process/data/abgcoqa/val.parquet \
    data.train_data_num=100 \
    data.val_data_num=100 \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.max_start_length=3072 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=0 \
    trainer.default_hdfs_dir=null \
    max_turns=4 \
    +ambigqa.enable_clarify_action=true \
    +ambigqa.max_clarify_turns=3 \
    +ambigqa.gpt4_simulator_url="http://10.10.211.118:8001/batch_generate" \
    +ambigqa.azure_openai_endpoint="https://cpii-s5.openai.azure.com/" \
    +ambigqa.azure_openai_api_key="91e5ea9bf61c4769a44b0b0b5c67d559" \
    +ambigqa.azure_openai_deployment="gpt-4o" \
    +ambigqa.azure_openai_api_version="2024-02-01" \
    +ambigqa.enable_entropy=false \
    +ipo.alpha=0.3 \
    +ipo.turn_cost=0.0 \
    +ipo.enable_ablation=true \
    +ipo.counterfactual_logprob=false \
    +ipo.efficiency_bonus=0.0 \
    +ipo.baseline_reward=0.0 \
    +ipo.clarify_bonus=0.15 \
    +ipo.ig_threshold=0.0 \
    +ipo.ambiguity_penalty=0.15 \
    +ipo.outcome_scale=1.0 \
    trainer.experiment_name=eval6-cross-eval-abgcoqa \
    2>&1 | tee exp6_cross_eval_abgcoqa.log

echo ""
echo "===== CROSS-EVAL SUMMARY ====="
echo "AmbigNQ results:"
grep -E "val/f1|val/clarify_rate|val/test_score" exp6_cross_eval_ambignq.log | tail -5
echo "AbgCoQA results:"
grep -E "val/f1|val/clarify_rate|val/test_score" exp6_cross_eval_abgcoqa.log | tail -5
echo "Job finished at: $(date)"
