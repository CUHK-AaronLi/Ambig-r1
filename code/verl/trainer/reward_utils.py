"""
Shared reward utilities for all reward manager variants.

Extracts common helpers from main_ppo.py:RewardManager so that
main_ppo_outcome_only, main_ppo_equalized, and main_ppo_ipo can
reuse them without code duplication.
"""

import re
import math
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch


class BaseRewardManager:
    """Base class providing shared decoding and scoring methods."""

    def __init__(self, tokenizer, num_examine, format_score=0., n_agent=1):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.n_agent = n_agent

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------
    def decode_response(self, data_item):
        """Common prompt/response decoding logic.

        Returns:
            dict with keys: prompt_ids, valid_prompt_ids, response_ids,
            valid_response_ids, valid_response_length, response_str,
            ground_truth, data_source, uid
        """
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
        uid = data_item.non_tensor_batch.get('uid', None)

        return {
            'prompt_ids': prompt_ids,
            'prompt_length': prompt_length,
            'valid_prompt_ids': valid_prompt_ids,
            'response_ids': response_ids,
            'valid_response_ids': valid_response_ids,
            'valid_response_length': valid_response_length,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'uid': uid,
        }

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_answer_text(text: str) -> str:
        ans = BaseRewardManager._extract_tag_content(text, 'answer')
        if ans:
            return ans
        action_pattern = re.compile(
            r"ACTION\s*:\s*\*\*ANSWER\*\*\s*:\s*(.*)",
            re.IGNORECASE | re.DOTALL,
        )
        match = action_pattern.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _extract_references(ground_truth: Any) -> List[str]:
        refs: List[str] = []

        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, np.ndarray):
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
    def _extract_tag_content(text: str, tag: str) -> str:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return ""
        return match.group(1).strip()

    # ------------------------------------------------------------------
    # Counting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_clarify_actions(text: str) -> int:
        count_tags = len(re.findall(
            r"<clarify>.*?</clarify>", text, flags=re.DOTALL | re.IGNORECASE
        ))
        count_action_line = len(re.findall(
            r"ACTION\s*:\s*\*\*CLARIFY\*\*", text, flags=re.IGNORECASE
        ))
        return count_tags + count_action_line

    # ------------------------------------------------------------------
    # Text normalization and tokenization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def _tokenize_for_eval(text: str):
        return re.findall(r"\w+|\S", text.lower())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _f1_score(prediction: str, ground_truth: str) -> float:
        pred_tokens = BaseRewardManager._tokenize_for_eval(prediction)
        gt_tokens = BaseRewardManager._tokenize_for_eval(ground_truth)
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
    def _max_f1(prediction: str, references) -> float:
        if not references:
            return 0.0
        scores = [
            BaseRewardManager._f1_score(prediction, ref)
            for ref in references
            if ref
        ]
        return max(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Turn boundary detection (for IPO and equalized reward modes)
    # ------------------------------------------------------------------
    @staticmethod
    def find_turn_boundaries(response_str: str) -> List[Tuple[str, int, int]]:
        """Find turn boundaries in a response string.

        Returns a list of (action_type, char_start, char_end) for each
        action block found. action_type is one of 'clarify', 'search', 'answer'.

        The char_start/char_end refer to the *closing tag* positions, which is
        where turn-level rewards should be placed.
        """
        boundaries = []
        for tag in ('clarify', 'search', 'answer'):
            pattern = re.compile(
                rf"<{tag}>(.*?)</{tag}>",
                re.DOTALL | re.IGNORECASE,
            )
            for match in pattern.finditer(response_str):
                # Record position of the closing tag
                close_tag = f"</{tag}>"
                close_start = match.end() - len(close_tag)
                boundaries.append((tag, close_start, match.end()))

        # Sort by position in the response
        boundaries.sort(key=lambda x: x[1])
        return boundaries

    @staticmethod
    def find_turn_token_positions(response_str: str, tokenizer) -> List[Tuple[str, int]]:
        """Find the token index of each turn's closing tag in the response.

        Returns a list of (action_type, token_index) pairs sorted by position.
        token_index is the index within the response token sequence where the
        closing tag ends.
        """
        boundaries = BaseRewardManager.find_turn_boundaries(response_str)
        results = []
        for action_type, char_start, char_end in boundaries:
            # Tokenize the prefix up to the end of the closing tag
            prefix = response_str[:char_end]
            n_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
            results.append((action_type, n_tokens - 1))  # -1 for 0-indexed
        return results

    # ------------------------------------------------------------------
    # Ablation scoring (IPO v3 counterfactual IG)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_observations(response_str: str) -> List[Dict]:
        """Parse response into turns with observations (search/clarify only).

        Returns list of dicts with keys:
            type, content, observation, char_start, char_end,
            obs_start, obs_end  (char positions of the observation block)
        """
        turns = []
        for tag in ('search', 'clarify', 'answer'):
            for match in re.finditer(
                rf'<{tag}>(.*?)</{tag}>', response_str, re.DOTALL | re.IGNORECASE
            ):
                turns.append({
                    'type': tag,
                    'content': match.group(1).strip(),
                    'char_start': match.start(),
                    'char_end': match.end(),
                    'observation': '',
                    'obs_start': -1,
                    'obs_end': -1,
                })
        turns.sort(key=lambda x: x['char_start'])

        for idx, turn in enumerate(turns):
            end = turn['char_end']
            next_start = turns[idx + 1]['char_start'] if idx + 1 < len(turns) else len(response_str)
            between = response_str[end:next_start]

            info_match = re.search(
                r'<information>(.*?)</information>', between, re.DOTALL | re.IGNORECASE
            )
            if info_match:
                turn['observation'] = info_match.group(1).strip()
                turn['obs_start'] = end + info_match.start()
                turn['obs_end'] = end + info_match.end()
                continue

            user_match = re.search(
                r'<user_response>(.*?)</user_response>', between, re.DOTALL | re.IGNORECASE
            )
            if user_match:
                turn['observation'] = user_match.group(1).strip()
                turn['obs_start'] = end + user_match.start()
                turn['obs_end'] = end + user_match.end()

        return turns

    @staticmethod
    def build_ablation_inputs(
        response_str: str,
        tokenizer,
        prompt_ids: torch.Tensor,
        references: List[str],
        answer_pred: str,
    ) -> Optional[List[Dict]]:
        """Build ablated inputs for counterfactual IG scoring.

        For each intermediate turn (search/clarify), constructs:
        - full context: prompt + response + best_ref_answer tokens
        - ablated context: prompt + response (with turn's observation removed) + best_ref_answer tokens

        The ref model will score the correct answer tokens given each context.
        Confidence delta = mean_log_prob(full) - mean_log_prob(ablated).

        Args:
            response_str: decoded response string
            tokenizer: HF tokenizer
            prompt_ids: 1D tensor of valid prompt token ids
            references: list of reference answer strings
            answer_pred: model's predicted answer string

        Returns:
            List of dicts with keys:
                turn_type, full_ids, ablated_ids, answer_start_full,
                answer_start_ablated, answer_token_len
            Returns None if no intermediate turns with observations.
        """
        if not references:
            return None

        # Select best reference answer (highest F1 with prediction)
        best_ref = max(references, key=lambda r: BaseRewardManager._f1_score(answer_pred, r))
        ref_tokens = tokenizer.encode(best_ref, add_special_tokens=False)
        if not ref_tokens:
            return None

        # Parse turns
        turns = BaseRewardManager._parse_observations(response_str)
        intermediate = [t for t in turns if t['type'] in ('search', 'clarify') and t['observation']]
        if not intermediate:
            return None

        # Encode full response once; full context is shared across all turns
        response_tokens = tokenizer.encode(response_str, add_special_tokens=False)
        prompt_list = prompt_ids.tolist()
        full_ids = prompt_list + response_tokens + ref_tokens
        answer_start_full = len(prompt_list) + len(response_tokens)

        results = []
        for turn in intermediate:
            # Build ablated response: remove the observation block
            if turn['obs_start'] >= 0 and turn['obs_end'] >= 0:
                ablated_response = (
                    response_str[:turn['obs_start']] +
                    response_str[turn['obs_end']:]
                )
            else:
                continue

            ablated_response_tokens = tokenizer.encode(ablated_response, add_special_tokens=False)
            ablated_ids = prompt_list + ablated_response_tokens + ref_tokens

            results.append({
                'turn_type': turn['type'],
                'full_ids': full_ids,
                'ablated_ids': ablated_ids,
                'answer_start_full': answer_start_full,
                'answer_start_ablated': len(prompt_list) + len(ablated_response_tokens),
                'answer_token_len': len(ref_tokens),
            })

        return results if results else None

