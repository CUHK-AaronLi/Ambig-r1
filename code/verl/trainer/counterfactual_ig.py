"""
Counterfactual Information Gain via Logprob Comparison.

For each sample with intermediate turns (clarify/search):
  baseline: prompt + "<think>direct answer</think><answer>ANSWER</answer>"
  informed: prompt + full_response (with observations)

  IG = mean_logP(answer | informed) - mean_logP(answer | baseline)

Positive IG = clarification helped, negative = hurt.
This replaces the text-overlap IG proxy with a true counterfactual measure.
"""

import re
import torch
import numpy as np
from verl import DataProto
from verl.trainer.reward_utils import BaseRewardManager


def compute_counterfactual_logprobs(trainer, batch: DataProto) -> np.ndarray:
    """Compute counterfactual IG for each sample in the batch.

    Args:
        trainer: RayPPOTrainer instance (has .tokenizer, .ref_policy_wg)
        batch: DataProto with responses and ref_log_prob already computed

    Returns:
        np.array[batch_size] of floats. None for samples without tools.
    """
    tokenizer = trainer.tokenizer
    batch_size = len(batch)
    results = np.empty(batch_size, dtype=object)

    # Collect baseline sequences for samples with intermediate turns
    baseline_info = []  # list of (sample_idx, baseline_ids, ans_offset_in_baseline_resp, ans_len, ans_offset_in_full_resp)

    dummy_rm = BaseRewardManager(tokenizer, num_examine=0)

    for i in range(batch_size):
        data_item = batch[i]
        decoded = dummy_rm.decode_response(data_item)
        response_str = decoded['response_str']

        # Check for intermediate turns
        turns = BaseRewardManager._parse_observations(response_str)
        intermediate = [t for t in turns if t['type'] in ('search', 'clarify')]

        if not intermediate:
            results[i] = None
            continue

        # Extract answer text
        match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL)
        if not match:
            results[i] = None
            continue

        answer_text = match.group(1).strip()
        if not answer_text:
            results[i] = None
            continue

        # Tokenize answer
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        if not answer_tokens:
            results[i] = None
            continue
        ans_len = len(answer_tokens)

        # --- Baseline: prompt + direct answer (no observations) ---
        direct_response = f"<think>\nThis question is clear and I can answer it directly.\n</think>\n<answer>{answer_text}</answer>"
        direct_resp_tokens = tokenizer.encode(direct_response, add_special_tokens=False)

        # Find answer offset in baseline response
        # Encode prefix up to answer content
        direct_prefix = f"<think>\nThis question is clear and I can answer it directly.\n</think>\n<answer>"
        direct_prefix_tokens = tokenizer.encode(direct_prefix, add_special_tokens=False)
        ans_offset_baseline = len(direct_prefix_tokens)

        # Get prompt ids
        prompt_ids = decoded['valid_prompt_ids']
        if isinstance(prompt_ids, torch.Tensor):
            prompt_list = prompt_ids.tolist()
        else:
            prompt_list = list(prompt_ids)

        baseline_ids = prompt_list + direct_resp_tokens

        # --- Find answer offset in FULL response ---
        # Encode response text up to "<answer>" tag + answer content start
        prefix_before_answer = response_str[:match.start(1)]
        prefix_tokens = tokenizer.encode(prefix_before_answer, add_special_tokens=False)
        ans_offset_full = len(prefix_tokens)

        baseline_info.append({
            'sample_idx': i,
            'baseline_ids': baseline_ids,
            'baseline_resp_len': len(direct_resp_tokens),
            'ans_offset_baseline': ans_offset_baseline,
            'ans_offset_full': ans_offset_full,
            'ans_len': ans_len,
            'n_turns': len(intermediate),
        })

    if not baseline_info:
        print("[Counterfactual-IG] No samples with intermediate turns")
        return results

    # --- Forward pass baselines through ref model ---
    n_real = len(baseline_info)

    # Pad for world_size divisibility
    world_size = trainer.actor_rollout_wg.world_size
    remainder = n_real % world_size
    pad_count = (world_size - remainder) if remainder != 0 else 0

    all_baseline_ids = [info['baseline_ids'] for info in baseline_info]
    all_resp_lens = [info['baseline_resp_len'] for info in baseline_info]

    for _ in range(pad_count):
        all_baseline_ids.append(all_baseline_ids[-1])
        all_resp_lens.append(all_resp_lens[-1])

    n_seqs = len(all_baseline_ids)
    max_seq_len = max(len(s) for s in all_baseline_ids)
    max_resp_len = max(all_resp_lens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Build tensors (left-padded)
    input_ids = torch.full((n_seqs, max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n_seqs, max_seq_len), dtype=torch.long)
    position_ids = torch.zeros((n_seqs, max_seq_len), dtype=torch.long)
    responses = torch.full((n_seqs, max_resp_len), pad_id, dtype=torch.long)

    for j, seq in enumerate(all_baseline_ids):
        seq_len = len(seq)
        offset = max_seq_len - seq_len
        input_ids[j, offset:] = torch.tensor(seq, dtype=torch.long)
        attention_mask[j, offset:] = 1
        position_ids[j, offset:] = torch.arange(seq_len, dtype=torch.long)

        # Response = last resp_len tokens
        resp_len = all_resp_lens[j]
        resp_offset = max_resp_len - resp_len
        responses[j, resp_offset:] = torch.tensor(seq[-resp_len:], dtype=torch.long)

    # Forward through ref model in micro-batches
    micro_batch_size = max(world_size, 8)  # at least world_size for chunking
    # Ensure micro_batch_size is divisible by world_size
    if micro_batch_size % world_size != 0:
        micro_batch_size = ((micro_batch_size // world_size) + 1) * world_size

    baseline_logprobs_all = []

    for mb_start in range(0, n_seqs, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, n_seqs)
        mb_size = mb_end - mb_start

        # Ensure divisible by world_size
        if mb_size % world_size != 0:
            # Pad the micro-batch
            extra = world_size - (mb_size % world_size)
            mb_end_padded = min(mb_end + extra, n_seqs)
            if mb_end_padded - mb_start < mb_size + extra:
                # Can't extend, skip padding (will be handled by repeating last)
                pass
            mb_end = mb_end_padded
            mb_size = mb_end - mb_start

        mb_data = DataProto.from_dict(tensors={
            'input_ids': input_ids[mb_start:mb_end],
            'attention_mask': attention_mask[mb_start:mb_end],
            'position_ids': position_ids[mb_start:mb_end],
            'responses': responses[mb_start:mb_end],
        })
        mb_data.meta_info['micro_batch_size'] = min(8, mb_size)
        mb_data.meta_info['temperature'] = 1.0

        ref_output = trainer.ref_policy_wg.compute_ref_log_prob(mb_data)
        ref_lp = ref_output.batch['ref_log_prob']  # (mb_size, max_resp_len)

        for j in range(min(mb_size, n_real - len(baseline_logprobs_all))):
            info_idx = len(baseline_logprobs_all)
            if info_idx >= n_real:
                break
            info = baseline_info[info_idx]

            # Answer tokens in baseline response: offset from right
            resp_len = info['baseline_resp_len']
            resp_offset = max_resp_len - resp_len
            ans_start = resp_offset + info['ans_offset_baseline']
            ans_end = ans_start + info['ans_len']

            if ans_end <= ref_lp.shape[1] and ans_start >= 0:
                ans_lps = ref_lp[j, ans_start:ans_end].float()
                baseline_lp = ans_lps.mean().item() if ans_lps.numel() > 0 else 0.0
            else:
                baseline_lp = 0.0

            baseline_logprobs_all.append(baseline_lp)

    # --- Extract informed logprobs from existing ref_log_prob ---
    ref_log_prob_full = batch.batch.get('ref_log_prob', None)

    ig_values = []
    for idx, info in enumerate(baseline_info):
        if idx >= len(baseline_logprobs_all):
            results[info['sample_idx']] = None
            continue

        baseline_lp = baseline_logprobs_all[idx]
        sample_i = info['sample_idx']

        if ref_log_prob_full is not None:
            ans_start = info['ans_offset_full']
            ans_end = ans_start + info['ans_len']
            resp_len_full = ref_log_prob_full.shape[1]

            if ans_end <= resp_len_full and ans_start >= 0:
                informed_lps = ref_log_prob_full[sample_i, ans_start:ans_end].float()
                informed_lp = informed_lps.mean().item() if informed_lps.numel() > 0 else 0.0
            else:
                informed_lp = 0.0
        else:
            informed_lp = 0.0

        ig = informed_lp - baseline_lp
        results[sample_i] = ig
        ig_values.append(ig)

    if ig_values:
        print(f"[Counterfactual-IG] {len(ig_values)}/{batch_size} samples, "
              f"mean_IG={np.mean(ig_values):.4f}, "
              f"pos={sum(1 for v in ig_values if v > 0)}, "
              f"neg={sum(1 for v in ig_values if v < 0)}")

    return results
