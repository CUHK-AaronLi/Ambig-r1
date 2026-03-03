"""
IPO Reward Manager — Information Gain Policy Optimization.

Core innovation: turn-level dense reward based on information gain proxy.
Distributes outcome credit across intermediate turns proportional to
estimated contribution.

Reward distribution:
  - Each intermediate turn (clarify/search) gets:
      ig_per_turn = alpha * F1 / n_intermediate_turns
    placed at the closing tag position (</clarify> or </search>)
  - F1 outcome reward at last valid token position.

IG proxy rationale: If the model clarified and got the answer right, the
clarification contributed to the information gain. If it clarified but
still got it wrong, the clarification didn't help (ig_per_turn ≈ 0
because F1 ≈ 0).

alpha (default 0.5) controls IG vs outcome balance.
"""

from verl import DataProto
import torch
from verl.trainer.reward_utils import BaseRewardManager


# Default IG balance coefficient
DEFAULT_ALPHA = 0.5


class IPORewardManager(BaseRewardManager):
    """Information Gain Policy Optimization — turn-level dense reward."""

    def __init__(self, tokenizer, num_examine, format_score=0., n_agent=1,
                 alpha=DEFAULT_ALPHA):
        super().__init__(tokenizer, num_examine, format_score, n_agent)
        self.alpha = alpha

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(batch_size):
            data_item = data[i]
            decoded = self.decode_response(data_item)

            response_str = decoded['response_str']
            ground_truth = decoded['ground_truth']
            data_source = decoded['data_source']
            valid_response_length = decoded['valid_response_length']

            answer_pred = self._extract_answer_text(response_str)
            references = self._extract_references(ground_truth)
            f1 = self._max_f1(answer_pred, references)

            # Place F1 outcome reward at last valid token
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = f1

            # Find intermediate turn boundaries (clarify and search, NOT answer)
            turn_positions = self.find_turn_token_positions(
                response_str, self.tokenizer
            )
            intermediate_positions = [
                (action_type, token_idx)
                for action_type, token_idx in turn_positions
                if action_type in ('clarify', 'search') and token_idx < valid_response_length
            ]

            n_intermediate = len(intermediate_positions)

            # Distribute IG reward across intermediate turns
            if n_intermediate > 0:
                ig_per_turn = self.alpha * f1 / n_intermediate
                for action_type, token_idx in intermediate_positions:
                    reward_tensor[i, token_idx] += ig_per_turn

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                ig_total = self.alpha * f1 if n_intermediate > 0 else 0.0
                print(
                    f"[IPO] F1={f1:.3f} intermediate_turns={n_intermediate} "
                    f"IG_total={ig_total:.3f} | {response_str[:200]}"
                )

        return reward_tensor


import ray
import hydra


def log_main(message):
    print(f"[Main-IPO] {message}")


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    log_main("Program started")

    if not ray.is_initialized():
        log_main("Initializing Ray cluster")
        env_vars = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}

        if hasattr(config.trainer, 'env_vars'):
            env_vars.update(config.trainer.env_vars)

        ray_init_kwargs = {
            'runtime_env': {'env_vars': env_vars}
        }
        if hasattr(config.trainer, 'num_cpus') and config.trainer.num_cpus is not None:
            ray_init_kwargs['num_cpus'] = config.trainer.num_cpus

        ray.init(**ray_init_kwargs)
        log_main("Ray cluster initialized")

    log_main("Starting main task")
    ray.get(main_task.remote(config))
    log_main("Main task completed")


@ray.remote
def main_task(config):
    log_main("Entered main task")

    from verl.utils.fs import copy_local_path_from_hdfs

    from pprint import pprint
    from omegaconf import OmegaConf
    log_main("Printing config")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    log_main(f"Downloading model checkpoint: {config.actor_rollout_ref.model.path}")
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    log_main(f"Checkpoint downloaded: {local_path}")

    log_main("Instantiating tokenizer")
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
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

    if config.reward_model.enable:
        log_main("Enabling reward model")
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    n_agent = config.actor_rollout_ref.rollout.n_agent if hasattr(config.actor_rollout_ref.rollout, 'n_agent') else 1

    # Read alpha from config if provided, otherwise use default
    alpha = DEFAULT_ALPHA
    if hasattr(config, 'ipo') and hasattr(config.ipo, 'alpha'):
        alpha = config.ipo.alpha
    print(f"[IPO] alpha={alpha}, n_agent={n_agent}")

    log_main("Creating IPO reward manager")
    reward_fn = IPORewardManager(
        tokenizer=tokenizer, num_examine=1, n_agent=n_agent, alpha=alpha
    )
    val_reward_fn = IPORewardManager(
        tokenizer=tokenizer, num_examine=2, n_agent=1, alpha=alpha
    )

    log_main("Creating resource pool manager")
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    log_main("Creating PPO trainer")
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )

    log_main("Initializing workers")
    trainer.init_workers()

    log_main("Starting training")
    trainer.fit()
    log_main("Training completed")


if __name__ == '__main__':
    main()
