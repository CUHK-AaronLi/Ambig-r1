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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]

# 添加日志函数
def log_step(message, step_info=""):
    """统一的日志格式"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] 🔍 {message} {step_info}")

def log_enter_function(func_name, **kwargs):
    """记录进入函数"""
    log_step(f"进入函数: {func_name}", f"参数: {kwargs}")

def log_exit_function(func_name, result=None):
    """记录退出函数"""
    log_step(f"退出函数: {func_name}", f"结果: {result}")

def log_simple(message):
    """简化的日志，用于减少冗余"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

def log_model_output(message, data=None):
    """简化的模型输出日志"""
    if data is not None:
        if isinstance(data, dict):
            # 只显示关键信息
            summary = {k: v for k, v in data.items() if k in ['loss', 'accuracy', 'score', 'reward']}
            log_simple(f"{message}: {summary}")
        else:
            log_simple(f"{message}: {type(data).__name__}")
    else:
        log_simple(message)


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    config: object = None  # 添加配置参数，用于获取num_cpus设置

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())
    if 'valid_clarify_stats' in batch.meta_info:
        metrics['env/number_of_valid_clarify'] = float(np.array(batch.meta_info['valid_clarify_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []
        all_test_batches = []  # collect for raw F1 computation

        clarify_url = None
        enable_clarify = False
        enable_search = True
        if hasattr(self.config, 'ambigqa'):
            clarify_url = getattr(self.config.ambigqa, 'gpt4_simulator_url', None)
            enable_clarify = bool(getattr(self.config.ambigqa, 'enable_clarify_action', False))
            enable_search = bool(getattr(self.config.ambigqa, 'enable_search_action', True))

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
            clarify_url=clarify_url,
            enable_clarify=enable_clarify,
            enable_search=enable_search,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                               non_tensor_batch_keys=['extra_info'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                all_test_batches.append(test_batch)
        else:
            for batch_dict in self.val_dataloader:  
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                               non_tensor_batch_keys=['extra_info'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                print(f"[DEBUG _validate] val batch shape: {test_gen_batch.batch['input_ids'].shape}")
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                    all_test_batches.append(test_batch)

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        # ---- Raw F1 + action distribution (independent of reward manager) ----
        try:
            raw_metrics = self._compute_raw_val_metrics(all_test_batches)
            metric_dict.update(raw_metrics)
        except Exception as e:
            print(f"[Val] Warning: raw F1 computation failed: {e}")

        return metric_dict

    def _compute_raw_val_metrics(self, all_test_batches):
        """Compute raw F1, answer extraction rate, and action distribution from val batches.

        These metrics are independent of the reward manager and directly comparable
        with competitor baselines (UserRL, A²Search, etc.).
        """
        from verl.trainer.reward_utils import BaseRewardManager

        per_source = {}  # data_source -> {f1s, answer_extracted, clarify, search, answer, total}

        for test_batch in all_test_batches:
            batch_size = len(test_batch)
            for i in range(batch_size):
                data_item = test_batch[i]
                # Decode response
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum())
                response_ids = data_item.batch['responses'][:valid_response_length]
                response_str = self.tokenizer.decode(response_ids)

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                data_source = data_item.non_tensor_batch.get('data_source', 'unknown')

                if data_source not in per_source:
                    per_source[data_source] = {
                        'f1s': [], 'answer_extracted': 0,
                        'clarify': 0, 'search': 0, 'answer': 0, 'total': 0,
                        'f1s_with_clarify': [], 'f1s_no_clarify': [],
                    }
                stats = per_source[data_source]
                stats['total'] += 1

                # Extract answer and compute raw F1
                answer_pred = BaseRewardManager._extract_answer_text(response_str)
                references = BaseRewardManager._extract_references(ground_truth)
                f1 = BaseRewardManager._max_f1(answer_pred, references)
                stats['f1s'].append(f1)

                # Check if <answer> tag was present
                if re.search(r'<answer>.*?</answer>', response_str, re.DOTALL | re.IGNORECASE):
                    stats['answer_extracted'] += 1

                # Count action types
                n_clarify = len(re.findall(r'<clarify>.*?</clarify>', response_str, re.DOTALL | re.IGNORECASE))
                n_search = len(re.findall(r'<search>.*?</search>', response_str, re.DOTALL | re.IGNORECASE))
                n_answer = len(re.findall(r'<answer>.*?</answer>', response_str, re.DOTALL | re.IGNORECASE))
                stats['clarify'] += n_clarify
                stats['search'] += n_search
                stats['answer'] += n_answer

                # Split F1 by whether sample clarified (for Post-Clarify F1)
                if n_clarify > 0:
                    stats['f1s_with_clarify'].append(f1)
                else:
                    stats['f1s_no_clarify'].append(f1)

        metrics = {}
        for ds, stats in per_source.items():
            n = stats['total']
            if n == 0:
                continue
            metrics[f'val/f1/{ds}'] = np.mean(stats['f1s'])
            metrics[f'val/answer_rate/{ds}'] = stats['answer_extracted'] / n
            total_actions = stats['clarify'] + stats['search'] + stats['answer']
            if total_actions > 0:
                metrics[f'val/clarify_rate/{ds}'] = stats['clarify'] / total_actions
                metrics[f'val/search_rate/{ds}'] = stats['search'] / total_actions
                metrics[f'val/answer_action_rate/{ds}'] = stats['answer'] / total_actions
            metrics[f'val/avg_clarify/{ds}'] = stats['clarify'] / n
            metrics[f'val/avg_search/{ds}'] = stats['search'] / n
            # Post-Clarify F1: F1 only on samples that clarified
            if len(stats['f1s_with_clarify']) > 0:
                metrics[f'val/post_clarify_f1/{ds}'] = np.mean(stats['f1s_with_clarify'])
                metrics[f'val/post_clarify_n/{ds}'] = len(stats['f1s_with_clarify'])
            if len(stats['f1s_no_clarify']) > 0:
                metrics[f'val/no_clarify_f1/{ds}'] = np.mean(stats['f1s_no_clarify'])
                metrics[f'val/no_clarify_n/{ds}'] = len(stats['f1s_no_clarify'])

        print(f"[Val Raw Metrics] {metrics}")
        return metrics

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        clarify_url = None
        enable_clarify = False
        enable_search = True
        if hasattr(self.config, 'ambigqa'):
            clarify_url = getattr(self.config.ambigqa, 'gpt4_simulator_url', None)
            enable_clarify = bool(getattr(self.config.ambigqa, 'enable_clarify_action', False))
            enable_search = bool(getattr(self.config.ambigqa, 'enable_search_action', True))

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
            clarify_url=clarify_url,
            enable_clarify=enable_clarify,
            enable_search=enable_search,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        log_step(f"开始训练循环 - 总epoch数: {self.config.trainer.total_epochs}")
        for epoch in range(self.config.trainer.total_epochs):
            log_step(f"开始Epoch {epoch + 1}/{self.config.trainer.total_epochs}")
            print(f"\n🎯 开始 Epoch {epoch + 1}/{self.config.trainer.total_epochs}")
            print(f"📈 当前全局步数: {self.global_steps}/{self.total_training_steps}")
            print("-" * 50)
            
            batch_count = 0
            log_step(f"Epoch {epoch + 1} 开始处理训练数据")
            log_step(f"训练数据加载器长度: {len(self.train_dataloader)}")
            
            for batch_dict in self.train_dataloader:
                batch_count += 1
                log_step(f"处理批次 {batch_count} - 全局步数: {self.global_steps}")
                print(f'\n🔄 Epoch {epoch + 1}, Step {self.global_steps}/{self.total_training_steps}, Batch {batch_count}')
                print(f"📊 批次大小: {len(batch_dict.get('input_ids', []))}")
                print(f"⏱️  开始时间: {datetime.datetime.now().strftime('%H:%M:%S')}")
                
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # Only repeat by n_agent in agent (do_search) path; non-agent path repeats by n later
                if self.config.do_search:
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation, include extra_info for clarify action
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'], 
                                    non_tensor_batch_keys=['extra_info'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        log_step("非搜索模式 - 直接生成序列")
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        log_step("搜索模式 - 开始LLM循环")
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()

                        # Agent path: n_agent already provides diversity (n rollouts per query).
                        # Do NOT repeat by n here — run_llm_loop returns 1 trajectory per agent copy.
                        # Non-agent path (above) uses vLLM's n for expansion.
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # ---- Counterfactual logprob IG for IPO ----
                    if getattr(self.config, 'ipo', None) and getattr(self.config.ipo, 'counterfactual_logprob', False):
                        if self.use_reference_policy:
                            with _timer('counterfactual_ig', timing_raw):
                                from verl.trainer.counterfactual_ig import compute_counterfactual_logprobs
                                cf_scores = compute_counterfactual_logprobs(self, batch)
                                batch.non_tensor_batch['counterfactual_ig_scores'] = cf_scores
                        else:
                            print("[Counterfactual-IG] WARNING: counterfactual_logprob=true but no ref policy, skipping")

                    # ---- Ablation scoring for IPO v3 ----
                    elif getattr(self.config, 'ipo', None) and getattr(self.config.ipo, 'enable_ablation', False):
                        if self.use_reference_policy:
                            with _timer('ablation', timing_raw):
                                ablation_scores = self._compute_ablation_scores(batch)
                                batch.non_tensor_batch['ablation_scores'] = ablation_scores
                        else:
                            print("[IPOv3-Ablation] WARNING: enable_ablation=true but no ref policy, skipping")

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        # Agent path uses n_agent for diversity; non-agent path uses n
                        effective_n = self.config.actor_rollout_ref.rollout.n_agent if self.config.do_search else self.config.actor_rollout_ref.rollout.n
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=effective_n)

                        # AReW critique: optionally modify advantages with step-level critique signals
                        if hasattr(self.config, 'arew') and getattr(self.config.arew, 'enable', False):
                            from verl.trainer.ppo.arew_critique import apply_arew_critique
                            arew_metrics = apply_arew_critique(batch, self.config.arew)
                            metrics.update(arew_metrics)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    log_step(f"检查Critic预热: {self.config.trainer.critic_warmup} <= {self.global_steps}")
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        log_step("开始更新Actor")
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        log_step("Actor更新完成")
                    else:
                        log_step(f"Critic预热中，跳过Actor更新 (需要 {self.config.trainer.critic_warmup} 步)")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                log_step(f"全局步数增加到: {self.global_steps}")
                
                # 添加训练步骤完成信息
                end_time = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"✅ Step {self.global_steps-1} 完成 - 结束时间: {end_time}")
                print(f"📊 当前指标: {list(metrics.keys())}")
                if 'actor_loss' in metrics:
                    print(f"🎭 Actor Loss: {metrics['actor_loss']:.4f}")
                if 'critic_loss' in metrics:
                    print(f"🎯 Critic Loss: {metrics['critic_loss']:.4f}")
                print("-" * 30)

                if self.global_steps >= self.total_training_steps:
                    log_step(f"达到总训练步数: {self.total_training_steps}")
                    print(f"\n🎉 训练完成！总共完成 {self.global_steps} 步")
                    
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
                    
            log_step(f"Epoch {epoch + 1} 完成")
            
        log_exit_function("fit", "所有epoch完成")
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })

        return batch, metrics

    def _compute_ablation_scores(self, batch: DataProto) -> np.ndarray:
        """Compute counterfactual IG via ablation scoring with the ref model.

        For each sample with intermediate turns (search/clarify), constructs
        ablated contexts (with each turn's observation removed) and scores
        the correct answer tokens under both full and ablated contexts.

        Returns:
            np.array of dtype=object, length=batch_size. Each element is either
            None (no intermediate turns) or a list of (turn_type, delta) tuples
            where delta = confidence_full - confidence_ablated.
        """
        from verl.trainer.reward_utils import BaseRewardManager

        batch_size = len(batch)
        results = np.empty(batch_size, dtype=object)

        # Collect all ablation pairs across the batch
        all_pairs = []   # (sample_idx, pair_idx, turn_type, full_ids, ablated_ids, ans_start_full, ans_start_abl, ans_len)
        sample_pair_counts = []

        for i in range(batch_size):
            data_item = batch[i]
            decoded = BaseRewardManager.decode_response(
                BaseRewardManager(self.tokenizer, num_examine=0), data_item
            )
            response_str = decoded['response_str']
            ground_truth = decoded['ground_truth']
            valid_prompt_ids = decoded['valid_prompt_ids']

            answer_pred = BaseRewardManager._extract_answer_text(response_str)
            references = BaseRewardManager._extract_references(ground_truth)

            ablation_inputs = BaseRewardManager.build_ablation_inputs(
                response_str=response_str,
                tokenizer=self.tokenizer,
                prompt_ids=valid_prompt_ids,
                references=references,
                answer_pred=answer_pred,
            )

            if ablation_inputs is None:
                results[i] = None
                sample_pair_counts.append(0)
                continue

            sample_pair_counts.append(len(ablation_inputs))
            for pair_idx, pair in enumerate(ablation_inputs):
                all_pairs.append((
                    i, pair_idx, pair['turn_type'],
                    pair['full_ids'], pair['ablated_ids'],
                    pair['answer_start_full'], pair['answer_start_ablated'],
                    pair['answer_token_len'],
                ))

        if not all_pairs:
            print("[IPOv3-Ablation] No intermediate turns found, skipping ablation scoring")
            return results

        # Build batched tensors for ref model forward pass
        # We need both full and ablated contexts → 2 * len(all_pairs) sequences
        all_seqs = []
        all_ans_lens = []
        for (_, _, _, full_ids, ablated_ids, _, _, alen) in all_pairs:
            all_seqs.append(full_ids)
            all_ans_lens.append(alen)
            all_seqs.append(ablated_ids)
            all_ans_lens.append(alen)

        # Pad n_seqs to be divisible by world_size (required by DataProto.chunk)
        world_size = self.actor_rollout_wg.world_size
        n_seqs_real = len(all_seqs)
        remainder = n_seqs_real % world_size
        if remainder != 0:
            pad_count = world_size - remainder
            for _ in range(pad_count):
                all_seqs.append(all_seqs[-1])  # duplicate last seq as padding
                all_ans_lens.append(all_ans_lens[-1])

        # Pad sequences to uniform length
        max_len = max(len(s) for s in all_seqs)
        pad_id = self.tokenizer.pad_token_id or 0
        n_seqs = len(all_seqs)

        input_ids = torch.full((n_seqs, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n_seqs, max_len), dtype=torch.long)
        position_ids = torch.zeros((n_seqs, max_len), dtype=torch.long)

        for j, seq in enumerate(all_seqs):
            seq_len = len(seq)
            # Right-align (left-pad) to match the model's convention
            offset = max_len - seq_len
            input_ids[j, offset:] = torch.tensor(seq, dtype=torch.long)
            attention_mask[j, offset:] = 1
            position_ids[j, offset:] = torch.arange(seq_len, dtype=torch.long)

        # For compute_ref_log_prob, "responses" = the portion we want log_probs for.
        # Must be RIGHT-aligned: answer tokens at the end, pad on the left,
        # so that logits[:, -max_ans_len-1:-1] aligns correctly with answers.
        max_ans_len = max(all_ans_lens)
        responses = torch.full((n_seqs, max_ans_len), pad_id, dtype=torch.long)
        for j, seq in enumerate(all_seqs):
            alen = all_ans_lens[j]
            # Right-align: pad on left, answer tokens on right
            offset = max_ans_len - alen
            responses[j, offset:] = torch.tensor(seq[-alen:], dtype=torch.long)

        # Process in micro-batches to avoid OOM
        micro_batch_size = 8
        all_log_probs = []

        for mb_start in range(0, n_seqs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, n_seqs)
            mb_input_ids = input_ids[mb_start:mb_end]
            mb_attention_mask = attention_mask[mb_start:mb_end]
            mb_position_ids = position_ids[mb_start:mb_end]
            mb_responses = responses[mb_start:mb_end]

            # Build DataProto for ref model
            mb_data = DataProto.from_dict(
                tensors={
                    'input_ids': mb_input_ids,
                    'attention_mask': mb_attention_mask,
                    'position_ids': mb_position_ids,
                    'responses': mb_responses,
                }
            )
            mb_data.meta_info = {
                'micro_batch_size': micro_batch_size,
                'temperature': 1.0,
            }

            # Forward pass through ref model (no_grad is a no-op for remote
            # calls but signals intent)
            ref_output = self.ref_policy_wg.compute_ref_log_prob(mb_data)
            # ref_log_prob shape: [mb_size, max_ans_len]
            mb_log_probs = ref_output.batch['ref_log_prob'].cpu().float()
            all_log_probs.append(mb_log_probs)

        all_log_probs = torch.cat(all_log_probs, dim=0)  # [n_seqs, max_ans_len]
        # Discard padding sequences
        all_log_probs = all_log_probs[:n_seqs_real]

        # Compute confidence deltas
        pair_idx_offset = 0
        for i in range(batch_size):
            n_pairs = sample_pair_counts[i]
            if n_pairs == 0:
                continue

            deltas = []
            for p in range(n_pairs):
                seq_idx = (pair_idx_offset + p) * 2  # full context index
                alen = all_ans_lens[seq_idx]  # full and ablated share the same answer length

                # Mean log prob over answer tokens (right-aligned, so last `alen` positions)
                ans_offset = max_ans_len - alen
                full_lp = all_log_probs[seq_idx, ans_offset:].mean().item()
                ablated_lp = all_log_probs[seq_idx + 1, ans_offset:].mean().item()

                turn_type = all_pairs[pair_idx_offset + p][2]
                delta = full_lp - ablated_lp
                deltas.append((turn_type, delta))

            results[i] = deltas
            pair_idx_offset += n_pairs

        # Log summary
        n_with_turns = sum(1 for r in results if r is not None)
        all_deltas = [d for r in results if r is not None for _, d in r]
        if all_deltas:
            mean_delta = np.mean(all_deltas)
            print(f"[IPOv3-Ablation] {n_with_turns}/{batch_size} samples scored, "
                  f"mean_delta={mean_delta:.4f}, n_pairs={len(all_deltas)}")
        else:
            print(f"[IPOv3-Ablation] No valid ablation pairs computed")

        return results
