# Copyright 2026 Yunxiang Li
#
# AReW (Advantage Reweighting) Critique for Clarify-R1
#
# Implements step-level informativeness critiques following:
#   Zou et al. (2026) "On Information Self-Locking in RL for Active Reasoning"
#
# Two critique types:
#   AS (Action Selection): Was the clarify/search action informative?
#   Multi-action bonus: Did the trajectory explore multiple action types?
#
# Usage:
#   generation.py: accumulate_step_critiques() after each step
#   ray_trainer.py: apply_arew_critique() after compute_advantage()
#
# Config (Hydra):
#   +arew.enable=true +arew.lambda_as=0.3 +arew.lambda_multi=0.05

import torch
import numpy as np
from typing import List, Dict, Any


# Patterns indicating uninformative clarify responses
_UNINFORMATIVE_PATTERNS = [
    "[system]",
    "unable to process",
    "question is clear",
    "question is straightforward",
    "no need to clarify",
    "don't need clarification",
    "doesn't need clarification",
    "just answer the question",
    "i think my question is clear",
    "my question is straightforward",
]


def init_critique_tracker(batch_size: int) -> Dict[str, Any]:
    """Initialize per-trajectory critique accumulators."""
    return {
        'clarify_quality': [0.0] * batch_size,
        'search_quality': [0.0] * batch_size,
        'n_clarify': [0] * batch_size,
        'n_search': [0] * batch_size,
    }


def accumulate_step_critiques(
    tracker: Dict[str, Any],
    next_obs: List[str],
    is_search: List[int],
    is_clarify: List[int],
    active_mask: torch.Tensor,
) -> Dict[str, Any]:
    """
    Accumulate critique signals for one generation step.

    AS Critique for clarify:
      +1 if simulator response is meaningful (>5 words, no system error)
      -1 if response is system error, boilerplate, or too short

    AS Critique for search:
      +1 if search returned substantial content (>100 chars)
      -0.5 if search returned little/no content
    """
    for i in range(len(next_obs)):
        if i >= len(active_mask) or not active_mask[i]:
            continue

        obs = next_obs[i] if i < len(next_obs) else ""
        obs_lower = (obs or "").lower().strip()

        if i < len(is_clarify) and is_clarify[i]:
            tracker['n_clarify'][i] += 1
            if not obs_lower or len(obs_lower.split()) < 3:
                tracker['clarify_quality'][i] -= 1.0
            elif any(p in obs_lower for p in _UNINFORMATIVE_PATTERNS):
                tracker['clarify_quality'][i] -= 1.0
            else:
                tracker['clarify_quality'][i] += 1.0

        if i < len(is_search) and is_search[i]:
            tracker['n_search'][i] += 1
            if not obs_lower or len(obs_lower) < 100:
                tracker['search_quality'][i] -= 0.5
            else:
                tracker['search_quality'][i] += 1.0

    return tracker


def finalize_critiques(tracker: Dict[str, Any]) -> Dict[str, list]:
    """
    Compute final per-trajectory critique scores.

    Returns:
      as_critique: [-1, +1] per trajectory (mean quality of tool actions)
      multi_action: 1.0 if trajectory used any tool, 0.0 otherwise
      n_clarify, n_search: raw counts
    """
    batch_size = len(tracker['clarify_quality'])
    as_critique = [0.0] * batch_size
    multi_action = [0.0] * batch_size

    for i in range(batch_size):
        signals = []

        if tracker['n_clarify'][i] > 0:
            signals.append(
                tracker['clarify_quality'][i] / max(tracker['n_clarify'][i], 1)
            )

        if tracker['n_search'][i] > 0:
            signals.append(
                tracker['search_quality'][i] / max(tracker['n_search'][i], 1)
            )

        if signals:
            as_critique[i] = float(np.clip(sum(signals) / len(signals), -1.0, 1.0))
            multi_action[i] = 1.0

    return {
        'as_critique': as_critique,
        'multi_action': multi_action,
        'n_clarify': tracker['n_clarify'],
        'n_search': tracker['n_search'],
    }


def apply_arew_critique(data, arew_config) -> dict:
    """
    Apply AReW critique to modify advantages IN-PLACE.

    Formula: Â_t = A_t + λ_as * u_as + λ_multi * u_multi

    Args:
        data: DataProto with batch['advantages'] and meta_info['arew_critiques']
        arew_config: Hydra config with lambda_as, lambda_multi

    Returns:
        dict of logged metrics
    """
    critique_data = data.meta_info.get('arew_critiques')
    if critique_data is None:
        return {}

    advantages = data.batch['advantages']
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:].float()

    lambda_as = getattr(arew_config, 'lambda_as', 0.3)
    lambda_multi = getattr(arew_config, 'lambda_multi', 0.05)

    as_critique = critique_data['as_critique']
    multi_action = critique_data['multi_action']

    bs = min(advantages.shape[0], len(as_critique))

    for i in range(bs):
        u_total = lambda_as * as_critique[i] + lambda_multi * multi_action[i]
        if u_total != 0:
            advantages[i] += u_total * response_mask[i]

    data.batch['advantages'] = advantages

    # Compute logging metrics
    n_with_tools = sum(1 for m in multi_action[:bs] if m > 0)
    active_critiques = [c for c in as_critique[:bs] if c != 0]
    mean_as = float(np.mean(active_critiques)) if active_critiques else 0.0
    n_positive = sum(1 for c in active_critiques if c > 0)
    n_negative = sum(1 for c in active_critiques if c < 0)

    metrics = {
        'arew/mean_as_critique': mean_as,
        'arew/n_with_tools': n_with_tools,
        'arew/n_positive_critique': n_positive,
        'arew/n_negative_critique': n_negative,
        'arew/total_clarify': sum(critique_data['n_clarify'][:bs]),
        'arew/total_search': sum(critique_data['n_search'][:bs]),
    }

    print(f"[AReW] λ_as={lambda_as:.2f} λ_multi={lambda_multi:.2f} | "
          f"tools={n_with_tools}/{bs} | as_critique={mean_as:+.3f} "
          f"(+{n_positive}/-{n_negative})")

    return metrics
