"""
IPO Reward Manager v3 — Counterfactual Information Gain.

Core innovation: per-turn IG measured by ablation scoring with the
ref model. For each intermediate turn, we compute the ref model's
confidence in the correct answer with vs without the turn's observation.
delta = confidence_full - confidence_ablated.

When ablation scores are available (from ray_trainer._compute_ablation_scores):
  IG(k) = alpha * max(delta_k, 0) * f1  (outcome-gated, non-negative)

When ablation scores are NOT available (fallback):
  IG(k) = alpha * max(relevance_k, marginal_k) * f1  (text-overlap proxy)

Reward distribution per sample:
  - Last token: F1 outcome reward
  - Each clarify/search turn: IG reward (min: baseline_reward)
  - No turns + F1 high: efficiency bonus
  - Invalid action penalty: -0.1 per invalid turn (prevents format collapse)
"""

import re
import math
from typing import List, Tuple, Dict

from verl import DataProto
import torch
from verl.trainer.reward_utils import BaseRewardManager


# Hyperparameters
DEFAULT_ALPHA = 0.5       # IG weight
BASELINE_REWARD = 0.05    # Small constant per turn (prevents zero-gradient)
EFFICIENCY_BONUS = 0.15   # Bonus for answering correctly without any turns
MAX_IG_PER_SAMPLE = 0.5   # Cap total IG
INVALID_PENALTY = -0.1    # Penalty per invalid action (prevents format collapse)
CONFIDENCE_BETA = 0.1     # Weight for answer confidence bonus


class IPORewardManager(BaseRewardManager):
    """Information Gain Policy Optimization v3 — counterfactual IG."""

    def __init__(self, tokenizer, num_examine, format_score=0., n_agent=1,
                 alpha=DEFAULT_ALPHA, turn_cost=0.0,
                 efficiency_bonus=EFFICIENCY_BONUS,
                 baseline_reward=BASELINE_REWARD,
                 clarify_bonus=0.0,
                 counterfactual_logprob=False,
                 efficiency_gating=False,
                 max_clarify_turns=3,
                 ig_threshold=0.0,
                 ambiguity_penalty=0.0):
        super().__init__(tokenizer, num_examine, format_score, n_agent)
        self.alpha = alpha
        self.turn_cost = turn_cost
        self.efficiency_bonus = efficiency_bonus
        self.baseline_reward = baseline_reward
        self.clarify_bonus = clarify_bonus
        self.counterfactual_logprob = counterfactual_logprob
        self.efficiency_gating = efficiency_gating
        self.max_clarify_turns = max_clarify_turns
        self.ig_threshold = ig_threshold
        self.ambiguity_penalty = ambiguity_penalty

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        has_log_probs = 'old_log_probs' in data.batch.keys()
        already_print_data_sources = {}

        # Check for counterfactual logprob IG scores from ray_trainer
        counterfactual_ig_scores = data.non_tensor_batch.get('counterfactual_ig_scores', None)
        use_counterfactual = counterfactual_ig_scores is not None and self.counterfactual_logprob
        counterfactual_count = 0

        # Check for ablation scores from ray_trainer
        ablation_scores = data.non_tensor_batch.get('ablation_scores', None)
        use_ablation = ablation_scores is not None and not use_counterfactual  # counterfactual takes priority
        ablation_used_count = 0
        fallback_count = 0

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

            # ============================================================
            # 1. F1 outcome reward at last valid token
            # ============================================================
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = f1

            # ============================================================
            # 2. Parse turns with their observations
            # ============================================================
            turns = self._parse_turns_with_observations(response_str)
            intermediate_turns = [
                t for t in turns if t['type'] in ('clarify', 'search')
            ]

            # ============================================================
            # 3. Compute per-turn IG (ablation or fallback)
            # ============================================================
            n_intermediate = len(intermediate_turns)
            ig_total = 0.0
            ig_details = []
            ig_source = "none"

            # Ambiguity-aware IG gating: disable IG for non-ambiguous questions
            is_ambig_sample = None
            if self.ambiguity_penalty > 0:
                extra_info = data_item.non_tensor_batch.get('extra_info', {})
                if isinstance(extra_info, dict):
                    is_ambig_sample = extra_info.get('_is_ambiguous', None)
                    if is_ambig_sample is None:
                        req_clari = extra_info.get('req_clari', None)
                        if req_clari is not None:
                            is_ambig_sample = bool(req_clari)

            # Skip IG computation for non-ambiguous questions (IG gated off)
            compute_ig = True
            if self.ambiguity_penalty > 0 and is_ambig_sample is False:
                compute_ig = False

            if n_intermediate > 0 and compute_ig:
                # Priority: counterfactual logprob > ablation > fallback
                if use_counterfactual and counterfactual_ig_scores[i] is not None:
                    # ---- Counterfactual logprob IG (true counterfactual) ----
                    ig_source = "counterfactual"
                    counterfactual_count += 1
                    total_ig = float(counterfactual_ig_scores[i])
                    # Clamp IG to reasonable range (can be negative = tool hurt)
                    total_ig = max(min(total_ig, 2.0), -1.0)
                    per_turn_ig = total_ig / n_intermediate

                    for turn in intermediate_turns:
                        # Scale by alpha, with F1 gating (same as ablation path)
                        ig_reward = self.alpha * per_turn_ig * f1
                        # Floor at negative cap (don't penalize too harshly)
                        ig_reward = max(ig_reward, -0.2)
                        if turn['type'] == 'clarify' and self.clarify_bonus > 0:
                            ig_reward += self.clarify_bonus
                        ig_details.append({
                            'type': turn['type'],
                            'delta': per_turn_ig,
                            'reward': ig_reward,
                        })
                        ig_total += ig_reward

                else:
                    # Try ablation scores first
                    sample_ablation = None
                    if use_ablation and ablation_scores[i] is not None:
                        sample_ablation = ablation_scores[i]  # list of (turn_type, delta)

                    if sample_ablation is not None and len(sample_ablation) > 0:
                        # ---- Ablation-based IG (v3 counterfactual) ----
                        ig_source = "ablation"
                        ablation_used_count += 1
                        for turn_type, delta in sample_ablation:
                            # Shift delta by threshold: only IG above threshold counts as positive
                            effective_delta = delta - self.ig_threshold
                            clamped_delta = max(effective_delta, -0.5)  # floor at -0.5
                            f1_effective = max(f1, 0.1)       # P0: F1 floor for hard cases
                            ig_reward = self.alpha * clamped_delta * f1_effective
                            # Floor at baseline (or -0.2 for negative)
                            ig_reward = max(ig_reward, min(self.baseline_reward, -0.2))
                            if turn_type == 'clarify' and self.clarify_bonus > 0:
                                ig_reward += self.clarify_bonus

                            ig_details.append({
                                'type': turn_type,
                                'delta': delta,
                                'reward': ig_reward,
                            })
                            ig_total += ig_reward
                    else:
                        # ---- Fallback: text-overlap proxy (v2 style) ----
                        ig_source = "fallback"
                        fallback_count += 1
                        cumulative_obs = ""
                        prev_confidence = 0.0

                        for turn in intermediate_turns:
                            obs = turn['observation']
                            individual_relevance = self._max_f1(obs, references) if obs else 0.0

                            cumulative_obs += " " + obs if obs else ""
                            curr_confidence = self._max_f1(cumulative_obs.strip(), references) if cumulative_obs.strip() else 0.0
                            marginal_gain = max(curr_confidence - prev_confidence, 0.0)
                            prev_confidence = curr_confidence

                            ig_raw = max(individual_relevance, marginal_gain)
                            ig_reward = self.alpha * ig_raw * f1
                            ig_reward = max(ig_reward, self.baseline_reward)
                            if turn['type'] == 'clarify' and self.clarify_bonus > 0:
                                ig_reward += self.clarify_bonus

                            ig_details.append({
                                'type': turn['type'],
                                'delta': ig_raw,
                                'reward': ig_reward,
                            })
                            ig_total += ig_reward

                # Cap total IG
                if ig_total > MAX_IG_PER_SAMPLE:
                    scale = MAX_IG_PER_SAMPLE / ig_total
                    for d in ig_details:
                        d['reward'] *= scale
                    ig_total = MAX_IG_PER_SAMPLE

                # Efficiency gating: scale IG by (max - n) / max
                # Fewer clarify turns → higher multiplier (PIR-style)
                if self.efficiency_gating and n_intermediate > 0:
                    eff = max(0.0, (self.max_clarify_turns - n_intermediate) / self.max_clarify_turns)
                    for d in ig_details:
                        d['reward'] *= eff
                    ig_total *= eff

                # Place IG rewards at turn token positions
                turn_positions = self.find_turn_token_positions(
                    response_str, self.tokenizer
                )
                intermediate_positions = [
                    (action_type, token_idx)
                    for action_type, token_idx in turn_positions
                    if action_type in ('clarify', 'search') and token_idx < valid_response_length
                ]
                for idx, (action_type, token_idx) in enumerate(intermediate_positions):
                    if idx < len(ig_details):
                        reward_tensor[i, token_idx] += ig_details[idx]['reward']

            elif f1 > 0 and valid_response_length > 0 and self.efficiency_bonus > 0:
                # No intermediate turns but answered correctly: efficiency bonus
                reward_tensor[i, valid_response_length - 1] += self.efficiency_bonus

            # ============================================================
            # 4. Turn cost penalty (penalize excessive clarify/search)
            # ============================================================
            if self.turn_cost > 0 and n_intermediate > 0 and valid_response_length > 0:
                turn_penalty = -self.turn_cost * n_intermediate
                reward_tensor[i, valid_response_length - 1] += turn_penalty

            # ============================================================
            # 4b. Ambiguity-aware reward (oracle or classifier-based)
            # ============================================================
            ambig_reward = 0.0
            if self.ambiguity_penalty > 0 and valid_response_length > 0:
                n_clarify = len([t for t in intermediate_turns if t['type'] == 'clarify'])
                if is_ambig_sample is True and n_clarify == 0:
                    # Should have clarified but didn't → penalty
                    ambig_reward = -self.ambiguity_penalty
                elif is_ambig_sample is False and n_clarify > 0:
                    # Shouldn't have clarified but did → penalty (IG already gated off above)
                    ambig_reward = -self.ambiguity_penalty

                if ambig_reward != 0:
                    reward_tensor[i, valid_response_length - 1] += ambig_reward

            # ============================================================
            # 5. Invalid action penalty
            # ============================================================
            n_invalid = self._count_invalid_actions(response_str)
            if n_invalid > 0 and valid_response_length > 0:
                penalty = INVALID_PENALTY * n_invalid
                reward_tensor[i, valid_response_length - 1] += penalty

            # ============================================================
            # 6. Answer confidence bonus (from log-probs)
            # ============================================================
            confidence = 0.0
            if has_log_probs and valid_response_length > 0 and CONFIDENCE_BETA > 0:
                confidence = self._compute_answer_confidence(
                    data, i, decoded['prompt_length'], valid_response_length, response_str
                )
                reward_tensor[i, valid_response_length - 1] += CONFIDENCE_BETA * confidence

            # ============================================================
            # Logging
            # ============================================================
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                detail_str = " ".join(
                    f"{d['type'][0]}:d={d.get('delta', 0):.3f}/r={d['reward']:.3f}"
                    for d in ig_details
                ) if ig_details else "none"
                tc_str = f" tc={-self.turn_cost * n_intermediate:.3f}" if self.turn_cost > 0 else ""
                cb_str = f" cb={self.clarify_bonus:.3f}" if self.clarify_bonus > 0 else ""
                ab_str = f" ambig={ambig_reward:.3f}" if ambig_reward != 0 else ""
                print(
                    f"[IPOv3] F1={f1:.3f} turns={n_intermediate} invalid={n_invalid} "
                    f"IG={ig_total:.3f}({ig_source}) conf={confidence:.3f}{tc_str}{cb_str}{ab_str} [{detail_str}] "
                    f"| {response_str[:200]}"
                )

        if use_counterfactual:
            print(f"[IPOv3] Counterfactual logprob stats: {counterfactual_count} scored, "
                  f"{batch_size - counterfactual_count} no-turns/no-score")
        elif use_ablation:
            print(f"[IPOv3] Ablation stats: {ablation_used_count} ablation, {fallback_count} fallback, "
                  f"{batch_size - ablation_used_count - fallback_count} no-turns")

        return reward_tensor

    # ------------------------------------------------------------------
    # Observation parsing (delegates to BaseRewardManager._parse_observations)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_turns_with_observations(response_str: str) -> List[Dict]:
        """Parse response into turns with their observations.

        Each turn = { type, content, observation, char_start, char_end, obs_start, obs_end }
        """
        return BaseRewardManager._parse_observations(response_str)

    # ------------------------------------------------------------------
    # Invalid action counting
    # ------------------------------------------------------------------
    @staticmethod
    def _count_invalid_actions(response_str: str) -> int:
        """Count invalid action attempts (format errors like (search> etc.)."""
        # Count "My previous action is invalid" messages from env
        n = len(re.findall(r'My previous action is invalid', response_str, re.IGNORECASE))
        return n

    # ------------------------------------------------------------------
    # Answer confidence from log-probs
    # ------------------------------------------------------------------
    def _compute_answer_confidence(self, data: DataProto, idx: int,
                                    prompt_length: int,
                                    valid_response_length: int,
                                    response_str: str) -> float:
        """Average log-prob of answer tokens → sigmoid → [0, 1] confidence."""
        match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL)
        if not match:
            return 0.0

        prefix_text = response_str[:match.start()] + '<answer>'
        answer_content = match.group(1)
        if not answer_content.strip():
            return 0.0

        n_prefix = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))
        n_answer = len(self.tokenizer.encode(answer_content, add_special_tokens=False))
        if n_answer == 0:
            return 0.0

        log_probs = data.batch['old_log_probs'][idx]
        end_idx = min(n_prefix + n_answer, valid_response_length)
        answer_lp = log_probs[n_prefix:end_idx].float()
        if answer_lp.numel() == 0:
            return 0.0

        avg_lp = answer_lp.mean().item()
        return float(1.0 / (1.0 + math.exp(-avg_lp - 1.0)))


# ==================================================================
# Training entry point (unchanged)
# ==================================================================
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

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

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

    # Read IPO hyperparameters from config
    alpha = DEFAULT_ALPHA
    turn_cost = 0.0
    efficiency_bonus = EFFICIENCY_BONUS
    baseline_reward = BASELINE_REWARD
    clarify_bonus = 0.0
    counterfactual_logprob = False
    efficiency_gating = False
    max_clarify_turns = 3
    ig_threshold = 0.0
    ambiguity_penalty = 0.0
    if hasattr(config, 'ipo'):
        if hasattr(config.ipo, 'alpha'):
            alpha = config.ipo.alpha
        if hasattr(config.ipo, 'turn_cost'):
            turn_cost = config.ipo.turn_cost
        if hasattr(config.ipo, 'efficiency_bonus'):
            efficiency_bonus = config.ipo.efficiency_bonus
        if hasattr(config.ipo, 'baseline_reward'):
            baseline_reward = config.ipo.baseline_reward
        if hasattr(config.ipo, 'clarify_bonus'):
            clarify_bonus = config.ipo.clarify_bonus
        if hasattr(config.ipo, 'counterfactual_logprob'):
            counterfactual_logprob = config.ipo.counterfactual_logprob
        if hasattr(config.ipo, 'efficiency_gating'):
            efficiency_gating = config.ipo.efficiency_gating
        if hasattr(config.ipo, 'max_clarify_turns'):
            max_clarify_turns = config.ipo.max_clarify_turns
        if hasattr(config.ipo, 'ig_threshold'):
            ig_threshold = config.ipo.ig_threshold
        if hasattr(config.ipo, 'ambiguity_penalty'):
            ambiguity_penalty = config.ipo.ambiguity_penalty
    print(f"[IPO] alpha={alpha}, turn_cost={turn_cost}, efficiency_bonus={efficiency_bonus}, "
          f"baseline_reward={baseline_reward}, clarify_bonus={clarify_bonus}, "
          f"counterfactual_logprob={counterfactual_logprob}, efficiency_gating={efficiency_gating}, "
          f"ig_threshold={ig_threshold}, ambiguity_penalty={ambiguity_penalty}, n_agent={n_agent}")

    log_main("Creating IPO reward manager")
    reward_fn = IPORewardManager(
        tokenizer=tokenizer, num_examine=1, n_agent=n_agent, alpha=alpha,
        turn_cost=turn_cost, efficiency_bonus=efficiency_bonus,
        baseline_reward=baseline_reward, clarify_bonus=clarify_bonus,
        counterfactual_logprob=counterfactual_logprob,
        efficiency_gating=efficiency_gating, max_clarify_turns=max_clarify_turns,
        ig_threshold=ig_threshold, ambiguity_penalty=ambiguity_penalty,
    )
    val_reward_fn = IPORewardManager(
        tokenizer=tokenizer, num_examine=2, n_agent=1, alpha=alpha,
        turn_cost=turn_cost, efficiency_bonus=efficiency_bonus,
        baseline_reward=baseline_reward, clarify_bonus=clarify_bonus,
        counterfactual_logprob=False,  # validation uses standard reward
        efficiency_gating=efficiency_gating, max_clarify_turns=max_clarify_turns,
        ig_threshold=ig_threshold, ambiguity_penalty=ambiguity_penalty,
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
