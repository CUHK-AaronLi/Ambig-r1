import torch
import re
from collections import defaultdict
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests


def _deep_convert_numpy(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return [_deep_convert_numpy(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_convert_numpy(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    paper_store_path: str = None  # StructNav: path to paper_store.json
    topk: int = 3
    # 澄清相关配置（已取消检索）
    clarify_url: str = "http://127.0.0.1:8001/batch_generate"
    enable_clarify: bool = True
    enable_search: bool = True  # Set False to make search actions invalid (for PACIFIC/Abg-CoQA)

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # # 初始化日志文件路径（轨迹暂不保存，注释以关闭落盘）
        # log_dir = os.path.join('logs', 'trajectories')
        # os.makedirs(log_dir, exist_ok=True)
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # mode = "val" if self.is_validation else "train"
        # self.traj_log_file = os.path.join(log_dir, f"traj_{mode}_{timestamp}.jsonl")
        # print(f"📁 轨迹日志将保存至: {self.traj_log_file}")
        # StructNav: load paper store for skim/read execution
        self.paper_store = {}
        paper_store_path = getattr(config, 'paper_store_path', None)
        if paper_store_path:
            import json
            with open(paper_store_path) as f:
                self.paper_store = json.load(f)
            print(f"[StructNav] Loaded {len(self.paper_store)} papers from {paper_store_path}")


    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at clarify or answer operation (search removed)."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # StructNav: add skim and read as stop tokens
        responses_str = [resp.split("</skim>")[0] + "</skim>"
                 if "</skim>" in resp
                 else resp.split("</read>")[0] + "</read>"
                 if "</read>" in resp
                 else resp.split("</search>")[0] + "</search>"
                 if '</search>' in resp
                 else resp.split('</clarify>')[0] + '</clarify>'
                 if '</clarify>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _prepare_clarify_metadata(self, rollings: DataProto) -> List[Dict[str, Any]]:
        """Prepare clarify metadata for user simulator (generic, dataset-agnostic).

        v2: Also extracts golden_answers as answer_hints for:
        - Answer leakage detection in simulator (Stengel-Eskin et al. 2025)
        - Better simulator prompts (can give contextual hints without revealing answer)
        """

        batch_size = rollings.batch['input_ids'].shape[0]
        extra_info_array = rollings.non_tensor_batch.get('extra_info', [])
        data_source_array = rollings.non_tensor_batch.get('data_source')
        reward_model_array = rollings.non_tensor_batch.get('reward_model', [])

        def _parse_extra_info(item):
            """Parse extra_info item: handles str (JSON), dict, and other types."""
            if isinstance(item, dict):
                return item
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    return parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, ValueError):
                    return {}
            if hasattr(item, 'item'):
                try:
                    result = item.item()
                    return result if isinstance(result, dict) else {}
                except Exception:
                    return {}
            return {}

        def _extract_extra(idx):
            if isinstance(extra_info_array, (list, tuple)):
                item = extra_info_array[idx] if idx < len(extra_info_array) else {}
                return _parse_extra_info(item)
            try:
                import numpy as np
                if isinstance(extra_info_array, np.ndarray):
                    if idx < len(extra_info_array):
                        return _parse_extra_info(extra_info_array[idx])
            except ImportError:
                pass
            if isinstance(extra_info_array, dict):
                return extra_info_array
            if isinstance(extra_info_array, str):
                return _parse_extra_info(extra_info_array)
            return {}

        def _extract_reward_model(idx):
            """Extract reward_model dict for a given index."""
            if isinstance(reward_model_array, (list, tuple)):
                item = reward_model_array[idx] if idx < len(reward_model_array) else {}
                return _parse_extra_info(item)
            try:
                import numpy as np
                if isinstance(reward_model_array, np.ndarray):
                    if idx < len(reward_model_array):
                        return _parse_extra_info(reward_model_array[idx])
            except ImportError:
                pass
            if isinstance(reward_model_array, dict):
                return reward_model_array
            if isinstance(reward_model_array, str):
                return _parse_extra_info(reward_model_array)
            return {}

        def _pick_value(source, idx):
            if source is None:
                return None
            if isinstance(source, (list, tuple)):
                return source[idx] if idx < len(source) else None
            if isinstance(source, torch.Tensor):
                if source.dim() == 0:
                    return source.item()
                if idx < source.shape[0]:
                    value = source[idx]
                    return value.item() if hasattr(value, 'item') else value
            return source

        metadata: List[Dict[str, Any]] = []
        for i in range(batch_size):
            sample_extra = _extract_extra(i) or {}
            data_source = _pick_value(data_source_array, i) or sample_extra.get('data_source') or 'generic'

            ambig_q = sample_extra.get('original_question') or sample_extra.get('gold_question') or ""
            unambig_q = (sample_extra.get('gold_question')
                         or sample_extra.get('unambiguous_question')
                         or sample_extra.get('disambiguated_question')
                         or ambig_q)
            # Map multiple context field names across datasets:
            #   Abg-CoQA: user_simulator_context, ShARC: snippet+scenario
            clarify_context = (sample_extra.get('clarify_context')
                               or sample_extra.get('user_simulator_context')
                               or sample_extra.get('context')
                               or '')
            # ShARC: combine snippet + scenario if no context found
            if not clarify_context:
                snippet = sample_extra.get('snippet', '')
                scenario = sample_extra.get('scenario', '')
                if snippet or scenario:
                    parts = []
                    if snippet:
                        parts.append(f"Rule: {snippet}")
                    if scenario:
                        parts.append(f"Scenario: {scenario}")
                    clarify_context = ' '.join(parts)

            # Extract golden answers for answer leakage detection
            answer_hints = []
            rm_data = _extract_reward_model(i)
            if isinstance(rm_data, dict):
                gt = rm_data.get('ground_truth', {})
                if isinstance(gt, dict):
                    target = gt.get('target', [])
                    if hasattr(target, 'tolist'):
                        answer_hints = target.tolist()
                    elif isinstance(target, (list, tuple)):
                        answer_hints = list(target)

            # IntentionGym: pass missing_details for persona-based simulator
            missing_details = sample_extra.get('missing_details', None)
            if missing_details is not None:
                missing_details = _deep_convert_numpy(missing_details)

            metadata.append(
                {
                    'ambiguous_question': ambig_q,
                    'unambiguous_question': unambig_q,
                    'data_source': data_source,
                    'clarify_context': clarify_context,
                    'answer_hints': answer_hints,
                    'missing_details': missing_details,
                }
            )
        return metadata

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        # 保留 non_tensor_batch 以确保 extra_info 在所有轮次中可用
        new_rollings.non_tensor_batch.update(rollings.non_tensor_batch)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            out_bs = output.batch['responses'].shape[0] if 'responses' in output.batch else 'N/A'
            print(f"[DEBUG _generate_with_gpu_padding] no-pad: input_bs={batch_size}, output_bs={out_bs}")
            return output
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        padded_active_batch.meta_info.update(active_batch.meta_info)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        out_bs = trimmed_batch['responses'].shape[0] if 'responses' in trimmed_batch else 'N/A'
        print(f"[DEBUG _generate_with_gpu_padding] padded: input_bs={batch_size}, padded_bs={batch_size + padding_size}, trimmed_bs={out_bs}")
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        # 为每个 step 跟踪不同的样例
        total_samples = gen_batch.batch['input_ids'].shape[0]
        
        # 初始化时显示所有样例的基本信息
        print(f"\n{'='*80}")
        print(f"🚀 开始处理 {total_samples} 个样例")
        extra_info_array = gen_batch.non_tensor_batch.get('extra_info')
        if extra_info_array is not None and len(extra_info_array) > 0:
            for i in range(min(1, total_samples)):  # 显示前3个样例的基本信息
                if i < len(extra_info_array):
                    extra_info = extra_info_array[i]
                    if isinstance(extra_info, str):
                        try:
                            extra_info = json.loads(extra_info)
                        except (json.JSONDecodeError, ValueError):
                            extra_info = {}
                    if not isinstance(extra_info, dict):
                        extra_info = {}
                    original_question = extra_info.get('original_question', 'N/A')
                    is_ambiguous = extra_info.get('_is_ambiguous', False)
                    print(f"📝 Sample {i}: {original_question[:50]}{'...' if len(original_question) > 50 else ''} (Ambiguous: {is_ambiguous})")
        print(f"{'='*80}")
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_clarify_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # 初始化轨迹记录（当前关闭落盘）
        trajectories = None

        # AReW critique tracker (always-on, near-zero overhead)
        from verl.trainer.ppo.arew_critique import init_critique_tracker, accumulate_step_critiques, finalize_critiques
        arew_tracker = init_critique_tracker(gen_batch.batch['input_ids'].shape[0])

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break       
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info.update(rollings.meta_info)
            # Agent loop: n_agent provides diversity, use n=1 per generate call
            rollings_active.meta_info['n_override'] = 1
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # 每个step跟踪一个不同的活跃样例
            active_indices = [i for i, is_active in enumerate(active_mask) if is_active]
            if active_indices:
                # 轮流选择不同的样例进行跟踪
                trace_sample_idx = active_indices[step % len(active_indices)]
                
                trace_response = responses_str[trace_sample_idx]
                trace_action = "clarify" if "<clarify>" in trace_response else "answer"
                
                # 获取样例的背景信息
                trace_extra_info = {}
                extra_info_array = rollings.non_tensor_batch.get('extra_info')
                if extra_info_array is not None and trace_sample_idx < len(extra_info_array):
                    trace_extra_info = extra_info_array[trace_sample_idx]
                    if isinstance(trace_extra_info, str):
                        try:
                            trace_extra_info = json.loads(trace_extra_info)
                        except (json.JSONDecodeError, ValueError):
                            trace_extra_info = {}
                    if not isinstance(trace_extra_info, dict):
                        trace_extra_info = {}

                trace_original_question = trace_extra_info.get('original_question', 'N/A')
                trace_is_ambiguous = trace_extra_info.get('_is_ambiguous', False)
                
                print(f"\n{'='*60}")
                print(f"📍 STEP {step + 1} - SAMPLE {trace_sample_idx}: {trace_action.upper()} Action")
                print(f"📝 Question: {trace_original_question[:60]}{'...' if len(trace_original_question) > 60 else ''}")
                print(f"🤔 Ambiguous: {trace_is_ambiguous}")
                print(f"🤖 Response: {trace_response}")
                print(f"{'─'*60}")
                
            # 打印简要统计
            active_count = len(active_indices)
            print(f"📊 Step {step + 1}: {active_count} active samples")
            
            # Execute in environment and process observations
            # 提取extra_data用于clarify action
            extra_data = None
            if self.config.enable_clarify:
                # 从rollings中提取extra_info，包含gold_question等信息
                extra_data = self._prepare_clarify_metadata(rollings)
            # StructNav: pass extra_info for paper_store lookup
            if hasattr(rollings, 'non_tensor_batch') and 'extra_info' in rollings.non_tensor_batch:
                self._current_extra_info = rollings.non_tensor_batch['extra_info']
            next_obs, dones, valid_action, is_search, is_clarify = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=True, extra_data=extra_data
            )
            
            # 记录轨迹（已关闭落盘/追踪）
            # if trajectories is not None:
            #     for i in range(total_samples):
            #         if i < len(active_mask) and active_mask[i].item():  # 转换为 Python bool
            #             trajectories[i].append({
            #                 'turn': step + 1,
            #                 'response': responses_str[i] if i < len(responses_str) else '',
            #                 'observation': next_obs[i] if i < len(next_obs) else '',
            #                 'done': bool(dones[i]) if i < len(dones) else False,
            #                 'valid': bool(valid_action[i]) if i < len(valid_action) else False,
            #                 'is_search': bool(is_search[i]) if i < len(is_search) else False,
            #                 'is_clarify': bool(is_clarify[i]) if i < len(is_clarify) else False
            #             })
            
            # 继续跟踪刚才选择的样例的环境执行结果
            if active_indices and trace_sample_idx < len(next_obs):
                trace_obs = next_obs[trace_sample_idx] if next_obs[trace_sample_idx] else "No observation"
                trace_done = dones[trace_sample_idx]
                trace_valid = valid_action[trace_sample_idx]
                trace_is_search = is_search[trace_sample_idx] if len(is_search) > trace_sample_idx else 0
                trace_is_clarify = is_clarify[trace_sample_idx]
                
                print(f"🎯 Environment Response for Sample {trace_sample_idx}:")
                print(f"   ✅ Valid: {trace_valid} | ❓ Clarify: {trace_is_clarify} | 🏁 Done: {trace_done}")
                if trace_obs and len(trace_obs) > 0:
                    print(f"   📄 Observation: {trace_obs[:150]}{'...' if len(trace_obs) > 150 else ''}")
                print(f"{'='*60}")
            
            # 打印动作执行摘要
            valid_count = sum(valid_action)
            done_count = sum(dones)
            print(f"📊 Step {step + 1} Summary: {valid_count}/{len(valid_action)} valid actions, {done_count} completed")
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_clarify_stats += torch.tensor(is_clarify, dtype=torch.int)

            # AReW: accumulate step-level critique signals
            arew_tracker = accumulate_step_critiques(arew_tracker, next_obs, is_search, is_clarify, active_mask)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
            print(f"✅ Step {step + 1} 完成 - 剩余活跃样例: {active_mask.sum().item()}")
            
        # final LLM rollout
        if active_mask.sum():
            print(f"🎯 开始最终生成轮次 - 活跃实例数: {active_mask.sum().item()}")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info.update(rollings.meta_info)
            # Agent loop: n_agent provides diversity, use n=1 per generate call
            rollings_active.meta_info['n_override'] = 1
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # 提取extra_data用于clarify action（final rollout）
            extra_data_final = None
            if self.config.enable_clarify:
                # 从rollings中提取extra_info，包含gold_question等信息
                extra_data_final = self._prepare_clarify_metadata(rollings)
            
            _, dones, valid_action, is_search, is_clarify = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False, extra_data=extra_data_final
            )

            # 记录最终轨迹（已关闭）
            # if trajectories is not None:
            #     for i in range(total_samples):
            #         if i < len(active_mask) and active_mask[i].item():  # 转换为 Python bool
            #             trajectories[i].append({
            #                 'turn': 'final',
            #                 'response': responses_str[i] if i < len(responses_str) else '',
            #                 'observation': '',
            #                 'done': bool(dones[i]) if i < len(dones) else False,
            #                 'valid': bool(valid_action[i]) if i < len(valid_action) else False,
            #                 'is_search': bool(is_search[i]) if i < len(is_search) else False,
            #                 'is_clarify': bool(is_clarify[i]) if i < len(is_clarify) else False
            #             })

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_clarify_stats += torch.tensor(is_clarify, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
            
            print(f"✅ 最终生成轮次完成")
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_clarify_stats'] = valid_clarify_stats.tolist()
        meta_info['arew_critiques'] = finalize_critiques(arew_tracker)
        
        # 显示所有样例的最终统计总结
        print(f"\n{'='*80}")
        print(f"🏁 批次处理完成 - 最终统计")
        print(f"📊 总样例数: {total_samples}")
        print(f"📈 平均轮次: {turns_stats.float().mean():.1f}")
        print(f"✅ 平均有效动作: {valid_action_stats.float().mean():.1f}")
        print(f"❓ 平均澄清次数: {valid_clarify_stats.float().mean():.1f}")
        
        # 显示前几个样例的详细统计
        print(f"\n📋 样例详细统计 (前{min(3, total_samples)}个):")
        for i in range(min(3, total_samples)):
            turns = turns_stats[i].item()
            valid_actions = valid_action_stats[i].item()
            clarifies = valid_clarify_stats[i].item()
            print(f"   Sample {i}: {turns}轮 | {valid_actions}动作 | {clarifies}澄清")
        print(f"{'='*80}")
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # 保存轨迹到文件（已禁用）
        # self._save_trajectories(trajectories, gen_batch.non_tensor_batch.get('extra_info'))
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    # def _save_trajectories(self, trajectories: List[List[Dict]], extra_info: List[Dict] = None):
    #     """保存轨迹到 JSONL 文件以便调试和查看交互过程。使用追加模式，整个运行过程写入同一个文件。"""
    #     try:
    #         with open(self.traj_log_file, 'a', encoding='utf-8') as f:  # 使用追加模式
    #             for i, traj in enumerate(trajectories):
    #                 if not traj:  # 跳过空轨迹
    #                     continue
    #                 data = {
    #                     'sample_idx': i,
    #                     'trajectory': traj
    #                 }
    #                 if extra_info and i < len(extra_info):
    #                     sample_info = extra_info[i]
    #                     data['original_question'] = sample_info.get('original_question', '')
    #                     data['gold_question'] = sample_info.get('gold_question', '')
    #                     data['_is_ambiguous'] = sample_info.get('_is_ambiguous', False)
    #                 
    #                 f.write(json.dumps(data, ensure_ascii=False) + '\n')
    #         
    #         print(f"✅ 已追加 {len([t for t in trajectories if t])} 条轨迹到日志文件")
    #     except Exception as e:
    #         print(f"⚠️ 保存轨迹文件失败: {e}")
    #         import traceback
    #         traceback.print_exc()

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=False, extra_data=None) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            extra_data: Extra data containing gold_question and other context information
            
        Returns:
            Tuple of (next_obs, dones, valid_action, is_search, is_clarify)
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search, is_clarify = [], [], [], [], []
        
        # 调试：检查长度对齐
        if extra_data is not None:
            print(f"[DEBUG] execute_predictions: len(predictions)={len(predictions)}, len(cur_actions)={len(cur_actions)}, len(extra_data)={len(extra_data)}")
            if len(extra_data) > 0:
                print(f"[DEBUG] execute_predictions: first extra_data keys: {list(extra_data[0].keys())}")
        
        # 搜索动作
        search_queries = []
        search_indices = []
        search_results_map = {}
        if do_search and self.config.search_url:
            for idx, (action, content) in enumerate(zip(cur_actions, contents)):
                if action == 'search':
                    search_queries.append(content)
                    search_indices.append(idx)
            if search_queries:
                search_results = self._batch_search(search_queries)
                for si, idx in enumerate(search_indices):
                    search_results_map[idx] = search_results[si] if si < len(search_results) else "No results found."

        # 澄清动作
        clarify_queries = []
        clarify_extra = []
        clarify_action_count = sum([1 for action in cur_actions if action == 'clarify'])
        print(f"[DEBUG] execute_predictions: clarify_action_count={clarify_action_count}, enable_clarify={self.config.enable_clarify}, has_extra_data={extra_data is not None}")
        
        if self.config.enable_clarify and extra_data:
            # 确保 extra_data 长度匹配
            if len(extra_data) != len(cur_actions):
                print(f"[DEBUG] Padding extra_data: {len(extra_data)} -> {len(cur_actions)}")
                if len(extra_data) < len(cur_actions):
                    extra_data = extra_data + [{}] * (len(cur_actions) - len(extra_data))
                else:
                    extra_data = extra_data[:len(cur_actions)]
            
            for idx, (action, content) in enumerate(zip(cur_actions, contents)):
                if action == 'clarify':
                    clarify_queries.append(content)
                    clarify_extra.append(extra_data[idx] if idx < len(extra_data) else {})
            
            print(f"[DEBUG] Collected {len(clarify_queries)} clarify queries")
            if clarify_queries:
                print(f"[DEBUG] First clarify query: '{clarify_queries[0][:100]}...'")
            
            clarify_results = self.batch_clarify(clarify_queries, clarify_extra) if clarify_queries else []
            print(f"[DEBUG] Got {len(clarify_results)} clarify results")
            assert len(clarify_results) == len(clarify_queries) if clarify_queries else True
        else:
            clarify_results = []
            if not self.config.enable_clarify:
                print(f"[DEBUG] Clarify disabled in config")
            if not extra_data:
                print(f"[DEBUG] No extra_data provided")

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                is_clarify.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    is_clarify.append(0)
                elif action == 'search':
                    if not self.config.enable_search:
                        # Search disabled — treat as invalid, guide to correct actions
                        next_obs.append(f'\nMy previous action is invalid. '
                            'If I want to clarify, I should put the question between <clarify> and </clarify>. '
                            'If I want to give the final answer, I should put the answer between <answer> and </answer>. '
                            'Let me try again.\n')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                        is_clarify.append(0)
                    elif i in search_results_map:
                        obs_text = search_results_map[i]
                        next_obs.append(f'\n\n<information>{obs_text}</information>\n\n')
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(1)
                        is_clarify.append(0)
                    else:
                        next_obs.append(f'\n\n<information>Search service not available.</information>\n\n')
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(1)
                        is_clarify.append(0)
                elif action == 'clarify':
                    if self.config.enable_clarify and clarify_results:
                        clarify_result = clarify_results.pop(0)
                        next_obs.append(f'\n\n<user_response>{clarify_result["response"]}</user_response>\n\n')
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(0)
                        is_clarify.append(1)
                    else:
                        next_obs.append(f'\n\n<user_response>clarify service not available</user_response>\n\n')
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(0)
                        is_clarify.append(1)
                elif action == 'skim':
                    # StructNav: execute skim on document
                    paper_structure = self._get_paper_structure(i) if hasattr(self, '_get_paper_structure') else {'paragraphs': []}
                    obs = self._execute_skim(contents[i], paper_structure)
                    next_obs.append(f'\n\n<observation>{obs}</observation>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                    is_clarify.append(0)
                elif action == 'read':
                    # StructNav: execute read on document
                    paper_structure = self._get_paper_structure(i) if hasattr(self, '_get_paper_structure') else {'paragraphs': []}
                    obs = self._execute_read(contents[i], paper_structure)
                    next_obs.append(f'\n\n<observation>{obs}</observation>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                    is_clarify.append(0)
                else:
                    invalid_msg = (
                        "\nMy previous action is invalid. "
                        "If I want to clarify, I should put the question between <clarify> and </clarify>. "
                        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                    next_obs.append(invalid_msg)
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    is_clarify.append(0)
            
        return next_obs, dones, valid_action, is_search, is_clarify

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        
        for prediction in predictions:
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            action = None
            content = ''

            tag_match = re.search(r'<(skim|read|search|clarify|answer)>(.*?)</\1>', prediction, re.DOTALL | re.IGNORECASE)
            if tag_match:
                action = tag_match.group(1).lower()
                content = tag_match.group(2).strip()
            else:
                # 兼容 ACTION : **ACTION** : content
                line_match = re.search(r'ACTION\s*:\s*\*\*(\w+)\*\*\s*:\s*(.*)', prediction, re.IGNORECASE | re.DOTALL)
                if line_match:
                    candidate = line_match.group(1).lower()
                    if candidate in {'skim', 'read', 'search', 'clarify', 'answer'}:
                        action = candidate
                        content = line_match.group(2).strip()
                    elif candidate == 'think':
                        action = None
                        content = ''

            actions.append(action)
            contents.append(content)

        return actions, contents



    def _get_paper_structure(self, sample_idx: int) -> dict:
        """Get document paragraphs for a sample from paper_store."""
        if not hasattr(self, '_current_extra_info') or not self._current_extra_info:
            return {'paragraphs': []}
        if sample_idx >= len(self._current_extra_info):
            return {'paragraphs': []}
        extra = self._current_extra_info[sample_idx]
        if isinstance(extra, str):
            import json
            extra = json.loads(extra)
        paper_id = extra.get('paper_id', '')
        if paper_id in self.paper_store:
            return self.paper_store[paper_id]
        return {'paragraphs': []}

    def _execute_skim(self, content: str, paper_structure: dict) -> str:
        """Execute skim action: return first sentence of each paragraph in range."""
        import re
        try:
            parts = content.split(',')
            start = int(parts[0].strip())
            end = int(parts[1].strip()) if len(parts) > 1 else start + 10
            paragraphs = paper_structure.get('paragraphs', [])
            lines = []
            for i in range(max(0, start), min(end, len(paragraphs))):
                para = paragraphs[i]
                first_sent = re.split(r'(?<=[.!?])\s+', para)[0]
                if len(first_sent.split()) > 30:
                    first_sent = ' '.join(first_sent.split()[:30]) + '...'
                lines.append(f'[{i}] {first_sent}')
            return '\n'.join(lines) if lines else '[No content in range]'
        except Exception as e:
            return f'[Skim error: {e}]'

    def _execute_read(self, content: str, paper_structure: dict) -> str:
        """Execute read action: return full text of paragraphs in range (max 5)."""
        try:
            parts = content.split(',')
            start = int(parts[0].strip())
            end = int(parts[1].strip()) if len(parts) > 1 else start + 3
            end = min(end, start + 5)  # max 5 paragraphs
            paragraphs = paper_structure.get('paragraphs', [])
            lines = []
            for i in range(max(0, start), min(end, len(paragraphs))):
                lines.append(f'[{i}] {paragraphs[i]}')
            return '\n\n'.join(lines) if lines else '[No content in range]'
        except Exception as e:
            return f'[Read error: {e}]'


    def _batch_search(self, queries: List[str]) -> List[str]:
        """Call retrieval server for batch search queries."""
        if not self.config.search_url:
            return ["Search service not configured."] * len(queries)
        try:
            response = requests.post(
                self.config.search_url,
                json={"queries": queries, "topk": self.config.topk},
            )
            response.raise_for_status()
            results = response.json().get("result", [])
            formatted = []
            for docs in results:
                if isinstance(docs, list):
                    snippets = []
                    for j, doc in enumerate(docs):
                        text = doc.get("document", {}).get("text", str(doc)) if isinstance(doc, dict) else str(doc)
                        snippets.append(f"[{j+1}] {text}")
                    formatted.append("\n".join(snippets))
                else:
                    formatted.append(str(docs))
            return formatted
        except Exception as e:
            print(f"[ERROR] Search service failed: {e}")
            return ["Search service error."] * len(queries)

    def batch_clarify(self, queries: List[str] = None, extra_data: List[Dict] = None) -> List[Dict]:
        """
        Batchified clarify for queries.
        Args:
            queries: clarification questions to call the clarify simulator
            extra_data: extra data containing gold_question and other context information
        Returns:
            clarify results which contain user responses
        """
        if not self.config.enable_clarify or not self.config.clarify_url:
            return []
        results = self._batch_clarify(queries, extra_data)['result']
        return results

    def _batch_clarify(self, queries, extra_data=None):
        """
        Call clarify simulator service
        """
        try:
            print(f"[DEBUG] _batch_clarify called: len(queries)={len(queries) if queries else 0}, len(extra_data)={len(extra_data) if extra_data else 0}")
            print(f"[DEBUG] clarify_url={self.config.clarify_url}, enable_clarify={self.config.enable_clarify}")
            
            # Build clarify request format, aligned with updated gpt_simulator.py expectations
            clarify_queries = []
            for i, query in enumerate(queries):
                metadata = extra_data[i] if extra_data and i < len(extra_data) else {}

                ambiguous_q = (metadata.get('ambiguous_question') or metadata.get('original_question') or "").strip()
                unambiguous_q = (metadata.get('unambiguous_question') or metadata.get('gold_question') or ambiguous_q).strip()
                context = (metadata.get('clarify_context') or metadata.get('context') or "").strip()

                # Fix: For AmbigNQ (no table context), use gold_question to help simulator answer
                data_source = str(metadata.get('data_source', 'generic'))
                if not context and data_source == 'ambignq' and unambiguous_q and unambiguous_q != ambiguous_q:
                    context = f'The user asked: \"{ambiguous_q}\" They specifically meant: \"{unambiguous_q}\"'

                # Extract answer_hints for leakage detection
                answer_hints = metadata.get('answer_hints', [])
                if answer_hints and not isinstance(answer_hints, list):
                    answer_hints = [str(answer_hints)]
                answer_hints = [str(a) for a in answer_hints if str(a).strip()]

                clarify_queries.append({
                    "question": ambiguous_q,
                    "clarification_question": str(query) if query else "",
                    "unambiguous_question": unambiguous_q,
                    "context": context,
                    "data_source": str(metadata.get('data_source', 'generic')),
                    "answer_hints": answer_hints if answer_hints else None,
                    "missing_details": metadata.get('missing_details', None),
                })

            print(f"[DEBUG] Sending {len(clarify_queries)} clarify queries to {self.config.clarify_url}")
            if clarify_queries:
                print(f"[DEBUG] First query sample: question='{clarify_queries[0].get('question', '')[:50]}...', clarification_question='{clarify_queries[0].get('clarification_question', '')[:50]}...'")
            
            response = requests.post(
                self.config.clarify_url,
                json={"queries": clarify_queries, "return_scores": False},
            )
            print(f"[DEBUG] Clarify service response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            print(f"[DEBUG] Clarify service returned {len(result.get('result', []))} results")
            return result
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Clarify service connection error: {e}")
            print(f"[ERROR] URL: {self.config.clarify_url}")
            import traceback
            traceback.print_exc()
            return {"result": [{"response": "Clarify service connection error. I need more information to answer your question."} for _ in (queries or [])]}
        except requests.exceptions.Timeout as e:
            print(f"[ERROR] Clarify service timeout: {e}")
            return {"result": [{"response": "Clarify service timeout. I need more information to answer your question."} for _ in (queries or [])]}
        except Exception as e:
            print(f"[ERROR] Clarify service failed: {e}")
            import traceback
            traceback.print_exc()
            return {"result": [{"response": "Clarify service failed. I need more information to answer your question."} for _ in (queries or [])]}

