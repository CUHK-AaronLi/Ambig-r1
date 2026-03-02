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
import math
from collections import Counter
import numpy as np
from typing import Any, Dict, List


def _select_rm_score_fn(data_source):
    # 保留旧逻辑以便向后兼容，但默认回落到 EM 计算
    supported = ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle',
                 'ambigqa', 'ambignq', 'abgcoqa', 'pacific']
    if data_source in supported:
        return qa_em.compute_score_em
    return qa_em.compute_score_em


class RewardManager():
    """The reward manager.

    Reward = F1 (答案质量)
           + adaptive_clarify_reward (根据效果奖惩 clarify)
           + confidence_bonus (log-prob 置信度)
           + consistency_bonus (同组多采样一致性)

    通过 reward_weights dict 控制各分量权重，设为 0 即关闭。
    """

    DEFAULT_WEIGHTS = {
        'f1': 1.0,
        'adaptive_clarify': 0.3,   # clarify 效果奖惩
        'confidence': 0.1,         # log-prob 置信度
        'consistency': 0.2,        # 同组采样一致性
    }

    def __init__(self, tokenizer, num_examine, format_score=0.,
                 enable_entropy=False, reward_weights=None, n_agent=1) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.enable_entropy = enable_entropy
        self.n_agent = n_agent
        self.w = {**self.DEFAULT_WEIGHTS, **(reward_weights or {})}

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        has_log_probs = 'old_log_probs' in data.batch.keys()

        # ---- Pass 1: 逐样本计算 F1、clarify、confidence ----
        per_sample = []
        already_print_data_sources = {}

        for i in range(batch_size):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')

            answer_pred = self._extract_answer_text(response_str)
            references = self._extract_references(ground_truth)
            f1 = self._max_f1(answer_pred, references)
            clarify_cnt = self._count_clarify_actions(response_str)

            confidence = 0.0
            if has_log_probs and valid_response_length > 0 and self.w['confidence'] != 0:
                confidence = self._compute_confidence(data, i, prompt_length, valid_response_length)

            per_sample.append({
                'f1': f1,
                'clarify_cnt': clarify_cnt,
                'confidence': confidence,
                'answer_pred': answer_pred,
                'valid_response_length': valid_response_length,
                'data_source': data_source,
                'response_str': response_str,
            })

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(response_str)

        # ---- Pass 2: 同组一致性 (consistency) ----
        consistency_scores = self._compute_consistency(per_sample)

        # ---- Pass 3: 汇总 reward ----
        for i, info in enumerate(per_sample):
            f1 = info['f1']
            clarify_cnt = info['clarify_cnt']
            confidence = info['confidence']
            consistency = consistency_scores[i]

            r_f1 = self.w['f1'] * f1
            r_clarify = self.w['adaptive_clarify'] * self._adaptive_clarify_reward(f1, clarify_cnt)
            r_confidence = self.w['confidence'] * confidence
            r_consistency = self.w['consistency'] * consistency

            total = r_f1 + r_clarify + r_confidence + r_consistency
            reward_tensor[i, info['valid_response_length'] - 1] = total

        return reward_tensor

    # ------------------------------------------------------------------
    # Component 1: Adaptive Clarify Reward
    # ------------------------------------------------------------------
    @staticmethod
    def _adaptive_clarify_reward(f1: float, clarify_cnt: int) -> float:
        """
        根据最终答案质量决定 clarify 的奖惩：
        - 答对了 (F1>=0.5) 且 clarify 过 → 说明 clarify 有用，给正向奖励
        - 答错了且 clarify 过 → clarify 没帮上忙，加重惩罚
        - 没 clarify 且答对了 → 效率 bonus
        - 没 clarify 且答错了 → 中性（不额外惩罚，因为 F1 本身已经低了）
        """
        if clarify_cnt == 0:
            return 0.2 if f1 >= 0.5 else 0.0
        if f1 >= 0.5:
            return 0.1 * min(clarify_cnt, 3)
        else:
            return -0.2 * clarify_cnt

    # ------------------------------------------------------------------
    # Component 2: Confidence (answer token 平均 log-prob)
    # ------------------------------------------------------------------
    def _compute_confidence(self, data: DataProto, idx: int, prompt_length: int,
                            valid_response_length: int) -> float:
        """
        用 response 部分的平均 log-prob 作为 confidence proxy。
        归一化到 [0, 1] 区间（sigmoid 映射），避免绝对值量纲问题。
        """
        log_probs = data.batch['old_log_probs'][idx]  # (response_length,)
        valid_lp = log_probs[:valid_response_length].float()
        if valid_lp.numel() == 0:
            return 0.0
        avg_lp = valid_lp.mean().item()
        return float(1.0 / (1.0 + math.exp(-avg_lp - 1.0)))

    # ------------------------------------------------------------------
    # Component 3: Self-Consistency across n_agent samples
    # ------------------------------------------------------------------
    def _compute_consistency(self, per_sample: List[dict]) -> List[float]:
        """
        同一个 question 的 n_agent 个采样里，答案越一致 → consistency 越高。
        一致性高 + 没 clarify = 好（不需要 clarify）
        一致性低 + 做了 clarify = 好（识别了不确定性）
        一致性低 + 没 clarify = 差（该问没问）
        """
        n = len(per_sample)
        scores = [0.0] * n

        if self.n_agent <= 1 or self.w['consistency'] == 0:
            return scores

        group_size = self.n_agent
        num_groups = n // group_size if group_size > 0 else 0

        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = per_sample[start:end]

            answers = [self._normalize_text(s['answer_pred']) for s in group]
            answer_counts = Counter(answers)
            majority_count = answer_counts.most_common(1)[0][1] if answer_counts else 0
            agreement = majority_count / group_size

            for j in range(group_size):
                idx = start + j
                did_clarify = group[j]['clarify_cnt'] > 0

                if agreement >= 0.8:
                    scores[idx] = 0.2 if not did_clarify else 0.0
                elif agreement <= 0.4:
                    scores[idx] = 0.2 if did_clarify else -0.2
                else:
                    scores[idx] = 0.0

        return scores

    def _compute_general_reward(self, response_str: str, ground_truth: Any, data_source: str) -> float:
        """
        保留旧接口兼容（仅在 val 或单独调用时使用）。
        """
        answer_pred = self._extract_answer_text(response_str)
        references = self._extract_references(ground_truth)

        acc = self._max_f1(answer_pred, references)
        clarify_cnt = self._count_clarify_actions(response_str)

        return float(acc - 0.1 * clarify_cnt)

    @staticmethod
    def _extract_references(ground_truth: Any) -> List[str]:
        """
        Robustly extract reference answers from ground_truth.
        Avoids truth-value checks on numpy arrays.
        """
        refs: List[str] = []

        def _to_list(x):
            import numpy as _np
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, _np.ndarray):
                return x.tolist()
            return [x]

        if isinstance(ground_truth, dict):
            for key in ('target', 'answers'):
                if key in ground_truth:
                    refs.extend(_to_list(ground_truth.get(key)))
        else:
            refs.extend(_to_list(ground_truth))

        return [str(r).strip() for r in refs if str(r).strip() != '']

    @staticmethod
    def _extract_answer_text(text: str) -> str:
        # 优先解析 <answer> 标签
        ans = RewardManager._extract_tag_content(text, 'answer')
        if ans:
            return ans
        # 支持 ACTION : **ANSWER** : xxx 格式
        action_pattern = re.compile(r"ACTION\s*:\s*\*\*ANSWER\*\*\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
        match = action_pattern.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _count_clarify_actions(text: str) -> int:
        count_tags = len(re.findall(r"<clarify>.*?</clarify>", text, flags=re.DOTALL | re.IGNORECASE))
        count_action_line = len(re.findall(r"ACTION\s*:\s*\*\*CLARIFY\*\*", text, flags=re.IGNORECASE))
        return count_tags + count_action_line

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def _tokenize_for_eval(text: str):
        return re.findall(r"\w+|\S", text.lower())

    @staticmethod
    def _f1_score(prediction: str, ground_truth: str) -> float:
        pred_tokens = RewardManager._tokenize_for_eval(prediction)
        gt_tokens = RewardManager._tokenize_for_eval(ground_truth)
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        pred_counts = Counter(pred_tokens)
        gt_counts = Counter(gt_tokens)
        common = sum((pred_counts & gt_counts).values())
        if common == 0:
            return 0.0
        precision = common / sum(pred_counts.values())
        recall = common / sum(gt_counts.values())
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _max_f1(prediction: str, references):
        if not references:
            return 0.0
        scores = [
            RewardManager._f1_score(prediction, ref)
            for ref in references
            if ref
        ]
        return max(scores) if scores else 0.0

    @staticmethod
    def _extract_tag_content(text: str, tag: str) -> str:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
        ref_tokens = RewardManager._tokenize_for_eval(reference)
        hyp_tokens = RewardManager._tokenize_for_eval(hypothesis)
        if not hyp_tokens:
            return 0.0
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref_tokens[i: i + n]) for i in range(max(ref_len - n + 1, 0)))
            hyp_ngrams = Counter(tuple(hyp_tokens[i: i + n]) for i in range(max(hyp_len - n + 1, 0)))
            overlap = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
            total = max(sum(hyp_ngrams.values()), 1)
            precision = overlap / total
            if precision == 0:
                precision = 1e-9
            precisions.append(math.log(precision))
        geo_mean = math.exp(sum(precisions) / max_n)
        if hyp_len == 0:
            bp = 0.0
        elif hyp_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / max(hyp_len, 1))
        return bp * geo_mean

    @staticmethod
    def _extract_expected_answers(ground_truth: Dict[str, Any]) -> List[str]:
        """Gather answer candidates from target list or clarification answers."""
        seen = set()
        collected: List[str] = []

        def _add_candidate(candidate: str):
            normalized = (candidate or "").strip()
            if normalized and normalized not in seen:
                collected.append(normalized)
                seen.add(normalized)

        for candidate in ground_truth.get('target') or []:
            _add_candidate(candidate)

        if collected:
            return collected

        for clar in ground_truth.get('clarification_answers') or []:
            answer_list = clar.get('all_original_answers') or []
            if not answer_list and clar.get('original_answer'):
                answer_list = [clar.get('original_answer')]
            for candidate in answer_list:
                _add_candidate(candidate)

        return collected

    def _compute_abgcoqa_score(self, response_str: str, ground_truth: dict, data_item) -> float:
        answer_text = self._extract_tag_content(response_str, 'answer')
        clarify_text = self._extract_tag_content(response_str, 'clarify')

        gt_answers = self._extract_expected_answers(ground_truth)
        answer_f1 = self._max_f1(answer_text, gt_answers) if (answer_text and gt_answers) else 0.0

        gt_ambiguity = ground_truth.get('ambiguity', 'non_ambiguous')
        gt_is_ambiguous = gt_ambiguity == 'ambiguous'
        pred_is_ambiguous = bool(clarify_text)

        tp = 1 if pred_is_ambiguous and gt_is_ambiguous else 0
        fp = 1 if pred_is_ambiguous and not gt_is_ambiguous else 0
        fn = 1 if (not pred_is_ambiguous) and gt_is_ambiguous else 0
        tn = 1 if (not pred_is_ambiguous) and (not gt_is_ambiguous) else 0

        def f1_from_confusion(tp_val, fp_val, fn_val, has_positive):
            if not has_positive:
                return 1.0
            if tp_val == 0:
                return 0.0
            precision = tp_val / (tp_val + fp_val)
            recall = tp_val / (tp_val + fn_val)
            return 2 * precision * recall / (precision + recall)

        amb_has_positive = gt_is_ambiguous or pred_is_ambiguous
        non_amb_has_positive = (not gt_is_ambiguous) or (not pred_is_ambiguous)

        amb_class_f1 = f1_from_confusion(tp, fp, fn, amb_has_positive)
        non_amb_class_f1 = f1_from_confusion(tn, fn, fp, non_amb_has_positive)
        ambiguity_macro_f1 = (amb_class_f1 + non_amb_class_f1) / 2

        clarify_bleu = 0.0
        if gt_is_ambiguous and clarify_text:
            reference_question = ground_truth.get('clarification_question', '')
            if reference_question:
                clarify_bleu = self._bleu_score(reference_question, clarify_text)

        # 当前只使用答案F1作为奖励
        total_score = answer_f1
        return float(max(min(total_score, 1.0), 0.0))
    

    


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

    n_agent = config.actor_rollout_ref.rollout.n_agent if hasattr(config.actor_rollout_ref.rollout, 'n_agent') else 1

    log_main("创建奖励管理器")
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1,
                              enable_entropy=enable_entropy, n_agent=n_agent)

    # validation 不做 consistency（只有 1 个采样），关闭 confidence（无 log_probs）
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=2,
                                  enable_entropy=enable_entropy, n_agent=1,
                                  reward_weights={'confidence': 0, 'consistency': 0})

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
