# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    elif data_source in ['ambigqa', 'ambignq']:  # 新增：支持AmbigQA数据集
        return qa_em.compute_score_em  # 暂时使用相同的评分函数，后续可以扩展
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., enable_entropy=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.enable_entropy = enable_entropy  # 新增：是否启用entropy计算

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # 新增：AmbigQA动作评估
            if data_source in ['ambigqa', 'ambignq']:
                score = self._compute_ambigqa_score(sequences_str, ground_truth, data_item)
            else:
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor

    def _compute_ambigqa_score(self, sequences_str: str, ground_truth: list, data_item) -> float:
        """计算AmbigQA的奖励分数：EM accuracy + 澄清惩罚 + 答案entropy"""
        import re
        import numpy as np
        
        # 1. 计算EM accuracy (主要分数) - 使用现有的EM score计算
        em_score = qa_em.compute_score_em(sequences_str, ground_truth)
        
        # 2. 检查澄清惩罚：如果没有ambiguous但问了clarify，-0.5
        clarify_penalty = 0.0
        has_clarify = bool(re.search(r'<clarify>.*?</clarify>', sequences_str, re.DOTALL))
        
        # 使用数据中的_is_ambiguous标签来判断是否ambiguous 
        # data_item 是单个实例，extra_info 应该直接是字典
        extra_info = data_item.non_tensor_batch.get('extra_info', {})
        if isinstance(extra_info, dict):
            is_ambiguous = extra_info.get('_is_ambiguous', False)
        else:
            # 如果 extra_info 不是字典，可能有其他结构，先输出调试信息
            print(f"[DEBUG] extra_info type: {type(extra_info)}, content: {extra_info}")
            is_ambiguous = False
        

        
        if has_clarify and not is_ambiguous:
            clarify_penalty = -0.2
        
        # 3. 计算最终答案的entropy (可选)
        entropy_score = 0.0
        if hasattr(self, 'enable_entropy') and self.enable_entropy:
            entropy_score = self._compute_answer_entropy_from_logits(data_item)
        
        # 4. 计算总分
        total_score = em_score + clarify_penalty + entropy_score
        
        # 确保分数在合理范围内
        return total_score
    
    def _compute_answer_entropy_from_logits(self, data_item) -> float:
        """基于logits计算答案的entropy，规避长度影响"""
        try:
            # 获取logits数据
            if 'logits' not in data_item.batch.keys():
                return 0.0
            
            logits = data_item.batch['logits']  # 假设这是logits数据
            
            # 只计算答案部分的entropy
            answer_start = self._find_answer_start_position(data_item)
            if answer_start == -1:
                return 0.0
            
            # 计算答案部分的entropy
            answer_entropies = []
            for i in range(answer_start, len(logits)):
                if i < len(logits):
                    # 使用logits计算entropy，避免长度影响
                    logit_tensor = torch.tensor(logits[i])
                    
                    # 计算softmax概率
                    probs = torch.softmax(logit_tensor, dim=-1)
                    
                    # 过滤掉概率为0的值（避免log(0)）
                    valid_probs = probs[probs > 1e-10]
                    if len(valid_probs) > 0:
                        # 计算entropy: -Σ(p_i * log(p_i))
                        entropy = -torch.sum(valid_probs * torch.log(valid_probs)).item()
                        answer_entropies.append(entropy)
            
            if not answer_entropies:
                return 0.0
            
            # 使用平均entropy，规避长度影响
            avg_entropy = sum(answer_entropies) / len(answer_entropies)
            
            # 归一化到0-0.2范围
            normalized_entropy = min(avg_entropy / 5.0, 0.2)  # 假设最大entropy约为5.0
            
            return normalized_entropy
            
        except Exception as e:
            print(f"Error computing entropy from logits: {e}")
            return 0.0
    

    


import ray
import hydra


def log_main(message):
    print(f"[Main] {message}")


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    log_main("程序启动")
    
    if not ray.is_initialized():
        log_main("初始化Ray集群")
        # this is for local ray cluster
        # 添加num_cpus参数限制CPU资源使用，避免占用所有CPU导致程序卡住
        env_vars = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}
        
        # 添加内存管理相关的环境变量
        if hasattr(config.trainer, 'env_vars'):
            env_vars.update(config.trainer.env_vars)
            log_main(f"设置环境变量: {config.trainer.env_vars}")
        
        ray_init_kwargs = {
            'runtime_env': {'env_vars': env_vars}
        }
        if hasattr(config.trainer, 'num_cpus') and config.trainer.num_cpus is not None:
            ray_init_kwargs['num_cpus'] = config.trainer.num_cpus
            log_main(f"设置Ray CPU核心数: {config.trainer.num_cpus}")
        
        ray.init(**ray_init_kwargs)
        log_main("Ray集群初始化完成")

    log_main("启动主任务")
    ray.get(main_task.remote(config))
    log_main("主任务完成")


@ray.remote
def main_task(config):
    log_main("进入主任务")
    
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    log_main("打印初始配置")
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    log_main(f"下载模型检查点: {config.actor_rollout_ref.model.path}")
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    log_main(f"模型检查点下载完成: {local_path}")

    # instantiate tokenizer
    log_main("实例化tokenizer")
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    log_main("tokenizer实例化完成")

    # define worker classes
    log_main("定义worker类")
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
        log_main("使用FSDP策略")

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
        log_main("使用Megatron策略")

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        log_main("启用奖励模型")
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # 新增：AmbigQA特定配置检查
    enable_entropy = False
    if hasattr(config, 'ambigqa') and config.ambigqa.enable_clarify_action:
        log_main(f"AmbigQA澄清动作已启用，最大轮数: {config.ambigqa.max_clarify_turns}")
        print(f"[AmbigQA] Clarify action enabled with max_turns: {config.ambigqa.max_clarify_turns}")
        if hasattr(config.ambigqa, 'gpt4_simulator_url'):
            print(f"[AmbigQA] GPT-4 simulator URL: {config.ambigqa.gpt4_simulator_url}")
        # 检查是否启用entropy计算
        if hasattr(config.ambigqa, 'enable_entropy'):
            enable_entropy = config.ambigqa.enable_entropy
            print(f"[AmbigQA] Entropy calculation: {'enabled' if enable_entropy else 'disabled'}")

    log_main("创建奖励管理器")
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, enable_entropy=enable_entropy)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=2, enable_entropy=enable_entropy)

    log_main("创建资源池管理器")
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    log_main("创建PPO训练器")
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    
    log_main("初始化workers")
    trainer.init_workers()
    
    log_main("开始训练")
    trainer.fit()
    log_main("训练完成")


if __name__ == '__main__':
    main()
