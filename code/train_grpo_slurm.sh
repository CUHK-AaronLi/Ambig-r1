#!/bin/bash

## ONLY CHANGE THE BELOW PARAMETERS
#SBATCH --job-name="ambig-r1-grpo-training"  # Your job name
#SBATCH --account=pgs  # TARGET CHARGING ACCOUNT (请根据您的团队修改：wu-team/philip-team/william-team/pgs)
#SBATCH --qos=low  # Specify the QOS (high/regular/low) - 根据训练时间选择
#SBATCH --partition=gemini  # TARGET PARTITION
#SBATCH -o out/%j-%x.out  # OUTPUT LOG PATH
#SBATCH -e out/%j-%x.err  # ERR LOG PATH
#SBATCH --time=07-00:00:00  # Max Job Running Time (7 days for regular QoS)
#SBATCH --gpus=4  # How many GPU requests for your job (根据训练脚本中的 n_gpus_per_node=4)
#SBATCH --mail-type=ALL  # email notification (options: NONE, BEGIN, END, FAIL, REQ)
#SBATCH --mail-user=1155124348@link.cuhk.edu.hk  # your email address (请修改为您的邮箱)
#SBATCH --nodelist=CPIIGPU-211-135  # Optional: Only if you want to run on specify node

## ONLY CHANGE THE ABOVE PARAMETERS

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

## DO NOT CHANGE the following parameters as the default parameters are already optimized on server side
##SBATCH --cpus-per-gpu=10 # DEFAULT assign 10 CPUS for each GPU
##SBATCH --mem-per-cpu=10000 # DEFAULT assign 10GB memory for each CPU

## It means 1 GPU task will be assigned 12 CPU Cores and 120GB RAM,
## 2 GPUs task will be assigned 24 CPU Cores and 240GB RAM,
## 4 GPUs task will be assigned 48 CPU Cores and 480GB RAM,
## 8 GPUs task will be assigned 96 CPU Cores and 960GB RAM,

## Put your bash script below for your job, etc including cd to your directory, conda activate your environment, ...
## NO NEED TO USE CUDA_VISIBLE_DEVICES as it is managed by SLURM

# show actual node in output file, useful for diagnostics
hostname
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

# Create output directory if it doesn't exist
mkdir -p out

# Enable conda
# 根据实际路径选择（从错误日志看，应该使用 anaconda3）
# source ~/miniconda3/bin/activate  # 如果使用 miniconda
source ~/anaconda3/bin/activate  # 使用 anaconda3
eval "$(conda shell.bash hook)"

# Activate your conda environment (请根据您的实际环境名称修改)
conda activate searchr1  # 请修改为您实际使用的 conda 环境名称

# Change to your code directory
cd /mnt/users_home/cpii.local/yli/Ambig-R1-new/code

# Set environment variables (移除 CUDA_VISIBLE_DEVICES，SLURM 会自动管理)
export DATA_DIR='scripts/data_process/data/pacific'
export WAND_PROJECT='Ambig-R1'
export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=baseline-pacific-grpo-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

# Fix wandb timeout issue - set offline mode or increase timeout
# Option 1: Use offline mode (recommended for compute nodes without internet)
export WANDB_MODE=offline
# Option 2: Increase timeout (uncomment if you want online mode)
# export WANDB_INIT_TIMEOUT=300

# Set Azure OpenAI environment variables for GPT Simulator
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"

# Start GPT4 Simulator service in background on the compute node
echo "Starting GPT4 Simulator service on port 8001..."
cd scripts
nohup python3 gpt_simulator.py > ../out/gpt_simulator_${SLURM_JOB_ID}.log 2>&1 &
SIMULATOR_PID=$!
cd ..

# Wait for simulator to be ready (check if port 8001 is listening)
echo "Waiting for GPT4 Simulator to be ready..."
for i in {1..60}; do
    # Try to check if service is responding using Python (more reliable)
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=1)" > /dev/null 2>&1; then
        echo "GPT4 Simulator is ready! (checked after $((i*2)) seconds)"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Warning: GPT4 Simulator may not be ready after 120 seconds, but continuing anyway..."
        echo "Check simulator logs: out/gpt_simulator_${SLURM_JOB_ID}.log"
    fi
    sleep 2
done

# Function to cleanup simulator on exit
cleanup() {
    echo "Stopping GPT4 Simulator (PID: $SIMULATOR_PID)..."
    kill $SIMULATOR_PID 2>/dev/null || true
    wait $SIMULATOR_PID 2>/dev/null || true
}
trap cleanup EXIT

# Run the training script
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.15 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.num_cpus=40 \
    max_turns=4 \
    +ambigqa.enable_clarify_action=true \
    +ambigqa.max_clarify_turns=3 \
    +ambigqa.gpt4_simulator_url="http://127.0.0.1:8001/batch_generate" \
    +ambigqa.azure_openai_endpoint="https://cpii-s5.openai.azure.com/" \
    +ambigqa.azure_openai_api_key="91e5ea9bf61c4769a44b0b0b5c67d559" \
    +ambigqa.azure_openai_deployment="gpt-4o" \
    +ambigqa.enable_entropy=false \
    2>&1 | tee $EXPERIMENT_NAME.log

echo "Job finished at: $(date)"

