#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick prompting-based evaluation on Abg-CoQA without any training.
"""

import argparse
import json
import re
from collections import Counter
from typing import List, Tuple

import datasets
import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_split(path: str, max_samples: int = None):
    data_files = {"data": path}
    dataset = datasets.load_dataset("parquet", data_files=data_files, split="data")
    if max_samples is not None:
        limit = min(max_samples, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def extract_answer_segment(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text.lower())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
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


def max_f1(prediction: str, references: List[str]) -> float:
    if not references:
        return 0.0
    return max(f1_score(prediction, ref) for ref in references if ref)


def parse_action(text: str) -> Tuple[str, str]:
    """
    Parse LLM output to identify action and its content.
    Returns (action, content) where action in {'answer', 'clarify'}.
    Defaults to 'answer' when tags missing.
    """
    if not text:
        return "answer", ""
    match = re.search(r"<(clarify|answer)>(.*?)</\1>", text, re.IGNORECASE | re.DOTALL)
    if match:
        action = match.group(1).lower()
        content = match.group(2).strip()
        return action, content
    return "answer", text.strip()


def call_clarify_service(
    clarify_url: str,
    clarification_question: str,
    original_question: str,
    gold_question: str,
) -> str:
    """
    Call clarify simulator endpoint (batch_generate style).
    """
    if clarify_url.endswith("/generate"):
        payload = {
            "question": original_question,
            "clarification_question": clarification_question,
            "context": f"User's actual intent: {gold_question or 'Not provided'}",
        }
    else:
        payload = {
            "queries": [
                {
                    "question": original_question,
                    "clarification_question": clarification_question,
                    "context": f"User's actual intent: {gold_question or 'Not provided'}",
                }
            ],
            "return_scores": False,
        }

    try:
        response = requests.post(clarify_url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "result" in data and isinstance(data["result"], list) and data["result"]:
                return data["result"][0].get("response", "")
            if "response" in data:
                return data.get("response", "")
    except Exception as exc:
        print(f"[Clarify] Warning: clarify service failed -> {exc}")
    return "I need more information to answer this question."


def main():
    parser = argparse.ArgumentParser(description="Prompt-only evaluation on Abg-CoQA.")
    parser.add_argument("--model_path", type=str, required=True, help="HF model name or local path")
    parser.add_argument("--split_path", type=str, required=True, help="Path to parquet split (e.g., data/abgcoqa/test.parquet)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_path", type=str, default=None, help="Optional JSONL file to save predictions")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--enable_clarify", action="store_true", help="Allow LLM to take clarify action before answering")
    parser.add_argument("--clarify_url", type=str, default="http://127.0.0.1:8001/batch_generate", help="Clarify simulator endpoint")
    parser.add_argument("--max_turns", type=int, default=3, help="Maximum dialogue turns when clarify is enabled")
    args = parser.parse_args()

    device = torch.device("cpu") if args.use_cpu or not torch.cuda.is_available() else torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.eval()
    model.to(device)

    dataset = load_split(args.split_path, args.max_samples)
    print(f"Loaded {len(dataset)} samples from {args.split_path}")

    all_scores = []
    output_file = open(args.output_path, "w", encoding="utf-8") if args.output_path else None

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for example in tqdm(dataset, desc="Evaluating"):
        prompt = example["prompt"][0]["content"]
        conversation = prompt
        raw_responses = []
        predicted_answer = ""

        max_turns = args.max_turns if args.enable_clarify else 1
        for turn in range(max_turns):
            inputs = tokenizer(conversation, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, **gen_kwargs)
            continuation_ids = generated[0, inputs.input_ids.shape[1]:]
            response = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            raw_responses.append(response)
            action, content = parse_action(response)

            conversation = f"{conversation.strip()}\n\n{response.strip()}\n\n"

            if action == "clarify" and args.enable_clarify:
                clarify_resp = call_clarify_service(
                    clarify_url=args.clarify_url,
                    clarification_question=content,
                    original_question=example["extra_info"]["original_question"],
                    gold_question=example["extra_info"].get("gold_question", ""),
                )
                conversation = f"{conversation}<user_response>{clarify_resp}</user_response>\n\n"
                continue

            # either answered or clarify disabled
            predicted_answer = extract_answer_segment(response if action == "answer" else content)
            break

        references = example["reward_model"]["ground_truth"]["target"]
        score = max_f1(predicted_answer, references)
        all_scores.append(score)

        if output_file:
            record = {
                "id": example.get("extra_info", {}).get("id", ""),
                "question": example["extra_info"]["original_question"],
                "prediction": predicted_answer,
                "references": references,
                "f1": score,
                "raw_response": raw_responses,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if output_file:
        output_file.close()

    mean_f1 = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\nFinished evaluation on {len(all_scores)} samples.")
    print(f"Average F1: {mean_f1:.4f}")


if __name__ == "__main__":
    main()

