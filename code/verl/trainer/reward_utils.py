"""
Shared reward utilities for all reward manager variants.

Extracts common helpers from main_ppo.py:RewardManager so that
main_ppo_outcome_only, main_ppo_equalized, and main_ppo_ipo can
reuse them without code duplication.
"""

import re
import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np


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
