#!/usr/bin/env python3
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
Preprocess the Abg-CoQA dataset to parquet format for direct answer generation (no clarify option).
This version creates prompts that directly ask for answers without the clarify action option.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set

import datasets

from verl.utils.hdfs_io import copy, makedirs


def _default_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(script_dir, "../../../data/AKBC2021-Abg-CoQA/abg-coqa")
    )


def normalize_answer(answer: Any) -> List[str]:
    """
    Normalize answer field into a list of strings for unified downstream usage.
    """
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(a).strip() for a in answer if str(a).strip()]
    normalized = str(answer).strip()
    return [normalized] if normalized else []


def extract_original_answers(clar_answer: Dict[str, Any] | None) -> List[str]:
    if not clar_answer:
        return []

    ordered_keys = sorted(
        key for key in clar_answer.keys() if key.startswith("org_ans")
    )
    collected: List[str] = []
    for key in ordered_keys:
        collected.extend(normalize_answer(clar_answer.get(key)))

    deduped: List[str] = []
    seen: Set[str] = set()
    for item in collected:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def format_history(history_turns: List[Dict[str, Any]]) -> str:
    if not history_turns:
        return "No previous turns."

    sorted_turns = sorted(
        history_turns, key=lambda item: item.get("turn_id", 0)
    )
    formatted_segments = []
    for turn in sorted_turns:
        turn_id = turn.get("turn_id", "N/A")
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()
        formatted_segments.append(
            f"Turn {turn_id} - Question: {question or '[empty]'}"
        )
        formatted_segments.append(
            f"Turn {turn_id} - Answer: {answer or '[empty]'}"
        )
    return "\n".join(formatted_segments)


def build_user_simulator_context(
    story: str,
    history_text: str,
    original_question: str,
) -> str:
    segments = [
        "User Question Context:",
        f"Passage:\n{story or '[No passage provided]'}",
        "",
        f"Conversation history:\n{history_text or 'No previous turns.'}",
        "",
        f"Original question:\n{original_question or '[missing question]'}",
    ]
    return "\n".join(segments).strip()


def make_prefix(example: Dict[str, Any], template_type: str) -> str:
    """
    Create prompt prefix for direct answer generation (no clarify option).
    """
    question = example["question"]
    story = example["story"]
    history_text = example["history_text"]

    if template_type == "base":
        prefix = f"""You are given a passage and a multi-turn conversation grounded in that passage. \
Use the passage and the conversation history to answer the current user question directly. \
You must conduct reasoning inside <think> and </think> every time you gather information. \
After reasoning, provide your final answer inside <answer> and </answer>.

Passage:
{story}

Conversation history:
{history_text}

Current question: {question}
"""
    else:
        raise NotImplementedError(f"Unsupported template_type: {template_type}")
    return prefix


def build_solution(
    example: Dict[str, Any], clarification_choice: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    target_turn = example.get("target_turn", {})
    clarification_turn = example.get("clarification_turn") or {}
    clarification_answers = clarification_turn.get("answers") or []

    def _format_clar_answer(ans: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "clar_answer": (ans.get("clr_ans") or "").strip(),
            "original_answer": (ans.get("org_ans") or "").strip(),
            "all_original_answers": extract_original_answers(ans),
        }

    if clarification_choice is not None:
        clar_answers_processed = [_format_clar_answer(clarification_choice)]
        target_values = extract_original_answers(clarification_choice)
    else:
        clar_answers_processed = [
            _format_clar_answer(ans) for ans in clarification_answers
    ]
        target_values = normalize_answer(target_turn.get("answer"))

    return {
        "target": target_values or normalize_answer(target_turn.get("answer")),
        "ambiguity": example.get("ambiguity", "unknown"),
        "clarification_question": (clarification_turn.get("question") or "").strip(),
        "clarification_answers": clar_answers_processed,
        "rationale": (target_turn.get("rationale") or "").strip(),
        "span": {
            "start": target_turn.get("span_start"),
            "end": target_turn.get("span_end"),
        },
    }


def convert_example(
    example: Dict[str, Any],
    split: str,
    idx: int,
    template_type: str,
    data_source: str,
) -> List[Dict[str, Any]]:
    story = (example.get("story") or "").strip()
    history_turns = example.get("history_turns") or []
    target_turn = example.get("target_turn") or {}
    clarification_turn = example.get("clarification_turn") or {}
    clarification_answers = clarification_turn.get("answers") or []
    is_ambiguous = (example.get("ambiguity") or "").lower() == "ambiguous"

    original_question = (target_turn.get("question") or "").strip()
    if original_question and original_question[-1] not in {"?", "！", "。", "."}:
        original_question += "?"

    history_text = format_history(history_turns)
    user_context_text = build_user_simulator_context(
        story=story or "[No passage provided]",
        history_text=history_text,
        original_question=original_question
        or "Please answer the user question.",
    )
    clarification_turn_question = (clarification_turn.get("question") or "").strip()

    prompt_text = make_prefix(
        {
            "question": original_question or "Please answer the user question.",
            "story": story or "[No passage provided]",
            "history_text": history_text,
        },
        template_type=template_type,
    )

    base_prompt = [
            {
                "role": "user",
                "content": prompt_text,
            }
    ]

    def _variant_extra_info(
        clarification_choice: Dict[str, Any] | None, clarification_index: int | None
    ) -> Dict[str, Any]:
        choice_answers = extract_original_answers(clarification_choice)
        extra = {
            "split": split,
            "index": idx,
            "id": example.get("id"),
            "source": example.get("source"),
            "turn_id": target_turn.get("turn_id"),
            "story": story,
            "history_turns": history_turns,
            "target_turn": target_turn,
            "ambiguity": example.get("ambiguity"),
            "original_question": original_question,
            "gold_question": example.get("gold_question", original_question),
            "clarification_choice_index": clarification_index,
            "clarification_choice_total": len(clarification_answers),
            "clarification_choice_text": (
                (clarification_choice or {}).get("clr_ans") or ""
            ).strip(),
            "clarification_choice_answers": choice_answers,
            "clarification_turn_question": clarification_turn_question,
            "_is_ambiguous": is_ambiguous,
            "user_simulator_context": user_context_text,
            "clarification_reference": None,
            "data_source": data_source,
        }

        if clarification_index is not None:
            extra["variant_id"] = f"{example.get('id')}::clar_{clarification_index+1:02d}"
        else:
            extra["variant_id"] = f"{example.get('id')}::base"
        return extra

    variants: List[Dict[str, Any]] = []
    if clarification_answers:
        for clar_idx, clar_answer in enumerate(clarification_answers):
            choice_answers = extract_original_answers(clar_answer)
            clarification_reference = {
                "context": user_context_text,
                "clarification_question": clarification_turn_question,
                "clarification_user_answer": (clar_answer.get("clr_ans") or "").strip(),
                "expected_answers": choice_answers,
                "story": story,
                "history_text": history_text,
                "variant_choice_index": clar_idx,
            }

            variants.append(
                {
                    "data_source": data_source,
                    "prompt": base_prompt,
                    "ability": "ambiguous-question-answering",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": build_solution(
                            example, clarification_choice=clar_answer
                        ),
                    },
                    "extra_info": {
                        **_variant_extra_info(clar_answer, clar_idx),
                        "clarification_reference": clarification_reference,
                    },
                }
            )
    else:
        variants.append(
            {
                "data_source": data_source,
                "prompt": base_prompt,
                "ability": "ambiguous-question-answering",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": build_solution(example),
                },
                "extra_info": _variant_extra_info(None, None),
            }
        )

    return variants


def load_raw_split(split_path: str) -> List[Dict[str, Any]]:
    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data") or []
    return data


def build_dataset(
    examples: List[Dict[str, Any]],
    split: str,
    template_type: str,
    data_source: str,
) -> List[Dict[str, Any]]:
    processed = []
    for idx, example in enumerate(examples):
        processed.extend(
            convert_example(
                example=example,
                split=split,
                idx=idx,
                template_type=template_type,
                data_source=data_source,
            )
        )
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Abg-CoQA dataset for direct answer generation (no clarify option)."
    )
    parser.add_argument("--data_dir", default=_default_data_dir())
    parser.add_argument("--local_dir", default="./data/abgcoqa_direct")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", type=str, default="base")
    parser.add_argument("--minitest_limit", type=int, default=50)
    parser.add_argument("--tinytest_limit", type=int, default=500)

    args = parser.parse_args()

    data_source = "abgcoqa"

    split_files = {
        "train": "coqa_abg_train.json",
        "val": "coqa_abg_val.json",
        "test": "coqa_abg_test.json",
    }

    raw_splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, file_name in split_files.items():
        split_path = os.path.join(args.data_dir, file_name)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Split file {file_name} not found under {args.data_dir}."
            )
        raw_splits[split_name] = load_raw_split(split_path)
        print(f"Loaded {split_name} split with {len(raw_splits[split_name])} examples.")

    train_processed = build_dataset(
        raw_splits["train"], "train", args.template_type, data_source
    )
    val_processed = build_dataset(
        raw_splits["val"], "val", args.template_type, data_source
    )
    test_processed = build_dataset(
        raw_splits["test"], "test", args.template_type, data_source
    )

    minitest_limit = min(args.minitest_limit, len(val_processed))
    tinytest_limit = min(args.tinytest_limit, len(val_processed))

    minitest_processed = val_processed[:minitest_limit]
    tinytest_processed = val_processed[:tinytest_limit]

    print("Processed dataset sizes:")
    print(f"  train: {len(train_processed)}")
    print(f"  val: {len(val_processed)}")
    print(f"  test: {len(test_processed)}")
    print(f"  minitest: {len(minitest_processed)} (limit={minitest_limit})")
    print(f"  tinytest: {len(tinytest_processed)} (limit={tinytest_limit})")

    train_dataset = datasets.Dataset.from_list(train_processed)
    val_dataset = datasets.Dataset.from_list(val_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)
    minitest_dataset = datasets.Dataset.from_list(minitest_processed)
    tinytest_dataset = datasets.Dataset.from_list(tinytest_processed)

    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    minitest_dataset.to_parquet(os.path.join(local_dir, "minitest.parquet"))
    tinytest_dataset.to_parquet(os.path.join(local_dir, "tinytest.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print(f"\n✅ Successfully generated direct answer dataset in {local_dir}")

