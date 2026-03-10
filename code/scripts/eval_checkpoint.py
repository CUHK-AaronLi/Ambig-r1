#!/usr/bin/env python3
"""
Standalone checkpoint evaluation script for Clarify-R1.

Runs greedy generation on val/test sets and computes:
- Raw F1 and EM (directly comparable with competitor baselines)
- Answer extraction rate
- Action distribution (clarify / search / answer)

Usage:
    python scripts/eval_checkpoint.py \
        --model_path verl_checkpoints/rl-ipo-v4/global_step_50 \
        --data_path scripts/data_process/data/ambignq_fewshot/tinytest.parquet \
        --output_path eval_results/ipo_v4_step50.json

    # Eval SFT baseline:
    python scripts/eval_checkpoint.py \
        --model_path verl_checkpoints/sft-clarify-warmup/global_step_330 \
        --data_path scripts/data_process/data/ambignq_fewshot/tinytest.parquet \
        --output_path eval_results/sft_baseline.json

Requirements:
    pip install vllm transformers pandas
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# F1 / EM scoring (same as reward_utils.py for consistency)
# ---------------------------------------------------------------------------
def _tokenize(text: str):
    return re.findall(r"\w+|\S", text.lower())


def _f1_score(pred: str, gt: str) -> float:
    pred_tok = _tokenize(pred)
    gt_tok = _tokenize(gt)
    if not pred_tok and not gt_tok:
        return 1.0
    if not pred_tok or not gt_tok:
        return 0.0
    common = sum((Counter(pred_tok) & Counter(gt_tok)).values())
    if common == 0:
        return 0.0
    p = common / len(pred_tok)
    r = common / len(gt_tok)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def _em_score(pred: str, gt: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0


def _max_f1(pred: str, refs) -> float:
    if not refs:
        return 0.0
    return max(_f1_score(pred, r) for r in refs if r)


def _max_em(pred: str, refs) -> float:
    if not refs:
        return 0.0
    return max(_em_score(pred, r) for r in refs if r)


def _extract_answer(text: str) -> str:
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: ACTION: **ANSWER**: ...
    m2 = re.search(r'ACTION\s*:\s*\*\*ANSWER\*\*\s*:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()


def _extract_references(gt) -> list:
    if isinstance(gt, dict):
        refs = []
        for k in ('target', 'answers'):
            v = gt.get(k)
            if v is not None:
                refs.extend(v if isinstance(v, (list, tuple)) else [v])
        return [str(r).strip() for r in refs if str(r).strip()]
    if isinstance(gt, (list, tuple)):
        return [str(r).strip() for r in gt if str(r).strip()]
    return [str(gt).strip()] if gt else []


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_eval_data(data_path: str, max_samples: int = None):
    """Load parquet data and return list of (prompt_text, ground_truth) pairs."""
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    print(f"Columns: {list(df.columns)}")

    samples = []
    for _, row in df.iterrows():
        # Build prompt from the data
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, str):
            import ast
            try:
                extra_info = ast.literal_eval(extra_info)
            except Exception:
                extra_info = {}

        gt = extra_info.get('reward_model', {}).get('ground_truth', None)
        if gt is None:
            gt = row.get('reward_model', {}).get('ground_truth', None) if isinstance(row.get('reward_model'), dict) else None

        # The prompt is already tokenized in parquet; we need the raw text
        # For now, we'll build prompts from the question field
        question = extra_info.get('question', row.get('question', ''))

        samples.append({
            'question': question,
            'ground_truth': gt,
            'extra_info': extra_info,
        })

    if max_samples:
        samples = samples[:max_samples]

    return samples


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_responses(model_path: str, samples: list, max_new_tokens: int = 2048,
                       tensor_parallel: int = 1):
    """Generate responses using vLLM for fast batch inference."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
    )

    # Build prompts using chat template
    prompts = []
    for s in samples:
        q = s['question']
        if not q:
            prompts.append("")
            continue

        messages = [{"role": "user", "content": q}]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = q
        prompts.append(prompt)

    params = SamplingParams(
        temperature=0.0,  # greedy
        max_tokens=max_new_tokens,
        stop=["</answer>"],  # stop after answer tag
    )

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, params)

    responses = []
    for out in outputs:
        text = out.outputs[0].text
        # If stopped at </answer>, add the closing tag back
        if not text.endswith("</answer>") and "<answer>" in text:
            text += "</answer>"
        responses.append(text)

    return responses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(samples: list, responses: list):
    """Compute metrics from samples and their generated responses."""
    results = {
        'f1_scores': [],
        'em_scores': [],
        'answer_extracted': 0,
        'clarify_count': 0,
        'search_count': 0,
        'answer_count': 0,
        'total': len(samples),
        'per_sample': [],
    }

    for sample, response in zip(samples, responses):
        refs = _extract_references(sample['ground_truth'])
        answer = _extract_answer(response)
        f1 = _max_f1(answer, refs)
        em = _max_em(answer, refs)

        has_answer_tag = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL | re.IGNORECASE))
        n_clarify = len(re.findall(r'<clarify>.*?</clarify>', response, re.DOTALL | re.IGNORECASE))
        n_search = len(re.findall(r'<search>.*?</search>', response, re.DOTALL | re.IGNORECASE))
        n_answer = len(re.findall(r'<answer>.*?</answer>', response, re.DOTALL | re.IGNORECASE))

        results['f1_scores'].append(f1)
        results['em_scores'].append(em)
        if has_answer_tag:
            results['answer_extracted'] += 1
        results['clarify_count'] += n_clarify
        results['search_count'] += n_search
        results['answer_count'] += n_answer

        results['per_sample'].append({
            'question': sample['question'][:200],
            'answer_pred': answer[:200],
            'refs': [r[:100] for r in refs[:3]],
            'f1': f1,
            'em': em,
            'n_clarify': n_clarify,
            'n_search': n_search,
            'response_len': len(response),
        })

    n = results['total']
    total_actions = results['clarify_count'] + results['search_count'] + results['answer_count']

    summary = {
        'n_samples': n,
        'f1_mean': sum(results['f1_scores']) / n if n else 0,
        'em_mean': sum(results['em_scores']) / n if n else 0,
        'answer_extraction_rate': results['answer_extracted'] / n if n else 0,
        'avg_clarify_per_sample': results['clarify_count'] / n if n else 0,
        'avg_search_per_sample': results['search_count'] / n if n else 0,
        'action_distribution': {
            'clarify': results['clarify_count'] / total_actions if total_actions else 0,
            'search': results['search_count'] / total_actions if total_actions else 0,
            'answer': results['answer_count'] / total_actions if total_actions else 0,
        },
    }

    return summary, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate Clarify-R1 checkpoint')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to evaluation parquet file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save JSON results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--tensor_parallel', type=int, default=1)
    parser.add_argument('--responses_file', type=str, default=None,
                        help='Pre-generated responses JSON (skip generation)')
    args = parser.parse_args()

    # Load data
    samples = load_eval_data(args.data_path, args.max_samples)

    if args.responses_file and os.path.exists(args.responses_file):
        print(f"Loading pre-generated responses from {args.responses_file}")
        with open(args.responses_file) as f:
            responses = json.load(f)
    else:
        responses = generate_responses(
            args.model_path, samples,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel=args.tensor_parallel,
        )

    # Evaluate
    summary, detailed = evaluate(samples, responses)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data:  {args.data_path} ({summary['n_samples']} samples)")
    print("=" * 60)
    print(f"  F1 (mean):              {summary['f1_mean']:.4f}")
    print(f"  EM (mean):              {summary['em_mean']:.4f}")
    print(f"  Answer extraction rate: {summary['answer_extraction_rate']:.4f}")
    print(f"  Avg clarify/sample:     {summary['avg_clarify_per_sample']:.2f}")
    print(f"  Avg search/sample:      {summary['avg_search_per_sample']:.2f}")
    ad = summary['action_distribution']
    print(f"  Action dist:            clarify={ad['clarify']:.3f}  search={ad['search']:.3f}  answer={ad['answer']:.3f}")
    print("=" * 60)

    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        output = {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'summary': summary,
            'per_sample': detailed['per_sample'],
        }
        with open(args.output_path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_path}")

        # Also save responses for reuse
        resp_path = args.output_path.replace('.json', '_responses.json')
        with open(resp_path, 'w') as f:
            json.dump(responses, f, ensure_ascii=False)
        print(f"Responses saved to {resp_path}")


if __name__ == '__main__':
    main()
