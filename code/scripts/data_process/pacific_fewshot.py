"""
预处理 PACIFIC 数据集为统一 parquet 格式（Few-Shot Prompt 版本）

基于 pacific.py，仅修改 make_prefix 以加入 few-shot 示例，
帮助模型在 RL 训练初期就能正确格式化 <clarify> / <answer> 标签。
"""

import os
import json
import argparse
from typing import List, Dict, Any

import datasets

from verl.utils.hdfs_io import copy, makedirs


def _default_data_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../data/PACIFIC/data/pacific")
    )


def make_prefix(question: str, context: str, template_type: str) -> str:
    if template_type == "fewshot":
        return (
            "You are a helpful agent that answers questions based on provided context.\n"
            "\n"
            "## Available Actions\n"
            "Each turn, output ONE action using the exact XML tags below:\n"
            "- Think: <think>your reasoning</think>\n"
            "- Clarify: <clarify>your question to the user</clarify>\n"
            "- Answer: <answer>your final answer</answer>\n"
            "\n"
            "You may combine Think with Clarify or Answer in the same turn.\n"
            "ALWAYS use the XML tags exactly as shown. Do NOT omit closing tags.\n"
            "\n"
            "## Strategy\n"
            "- If the question is ambiguous or has multiple valid interpretations, "
            "use CLARIFY to ask the user before answering.\n"
            "- If the question is clear, use ANSWER directly.\n"
            "\n"
            "## Example 1: Ambiguous question → Clarify → Answer\n"
            "\n"
            "Question: What was the growth rate?\n"
            "Context: Revenue: 2020 $50M, 2021 $65M, 2022 $80M.\n"
            "\n"
            "<think>The question doesn't specify which period. "
            "It could be 2020-2021 or 2021-2022. I should ask.</think>\n"
            "<clarify>Which period's growth rate are you asking about?</clarify>\n"
            "\n"
            "<user_response>From 2021 to 2022.</user_response>\n"
            "\n"
            "<think>Growth from $65M to $80M is $15M, about 23.1%.</think>\n"
            "<answer>23.1%</answer>\n"
            "\n"
            "## Example 2: Clear question → Answer directly\n"
            "\n"
            "Question: What was the revenue in 2022?\n"
            "Context: Revenue: 2020 $50M, 2021 $65M, 2022 $80M.\n"
            "\n"
            "<think>The question clearly asks for 2022 revenue.</think>\n"
            "<answer>$80M</answer>\n"
            "\n"
            "---\n"
            "Now answer the following question. "
            "Use <clarify>...</clarify> if ambiguous, "
            "or <answer>...</answer> when confident.\n"
            "\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
        )
    elif template_type == "base":
        return (
            "You are a helpful agent. Your goal is to accurately answer questions from a user. "
            "You must balance the need to ask clarifying questions against the need to answer directly. "
            "Available Actions per turn: THINK, CLARIFY, ANSWER. "
            "THINK: Output your internal reasoning about the question (no asking, no final answer). "
            "CLARIFY: Ask one concise question to clarify the user's original question. "
            "ANSWER: Provide a final, direct answer based on your understanding. "
            "Rules for ANSWER: it must be final, direct, and not end with a question. "
            "If confidence is low but you skip clarification, you may still ANSWER. "
            "Core strategy: THINK to judge clarity; CLARIFY only when ambiguity would change the answer; "
            "ANSWER when reasonably confident to avoid unnecessary back-and-forth. "
            "Response format (one line): ACTION : **<ACTION_NAME>** : <your output>. "
            "THINK uses <think> </think> to wrap reasoning. "
            "CLARIFY uses the same one-line format; wrap the clarification question inside <clarify> </clarify> "
            "ANSWER uses the same one-line format; wrap the final answer inside <answer> </answer> and do not end with a question. "
            f"Question: {question} Context: {context}\n"
        )
    else:
        raise NotImplementedError(f"Unsupported template_type: {template_type}")


def normalize_answer(ans: Any) -> List[str]:
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(a).strip() for a in ans if str(a).strip()]
    return [str(ans).strip()]


def format_table(table_obj: Dict[str, Any]) -> str:
    if not table_obj:
        return ""
    rows = table_obj.get("table") or []
    text_rows = []
    for row in rows:
        cells = [str(cell).strip() for cell in row if str(cell).strip()]
        if cells:
            text_rows.append(" | ".join(cells))
    return "\n".join(text_rows)


def format_paragraphs(paragraphs: List[Dict[str, Any]]) -> str:
    if not paragraphs:
        return ""
    paragraphs = sorted(paragraphs, key=lambda x: x.get("order", 0))
    texts = [p.get("text", "").strip() for p in paragraphs if p.get("text")]
    return "\n".join(texts)


def format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for turn_id, turn in enumerate(history, start=1):
        q = turn.get("question", "").strip()
        a_list = normalize_answer(turn.get("answer"))
        a_text = "; ".join(a_list)
        lines.append(f"Q{turn_id}: {q}")
        lines.append(f"A{turn_id}: {a_text}")
    return "\n".join(lines)


def build_context(entry: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    parts = []
    table_text = format_table(entry.get("table") or {})
    if table_text:
        parts.append(f"Table:\n{table_text}")

    paragraph_text = format_paragraphs(entry.get("paragraphs") or [])
    if paragraph_text:
        parts.append(f"Paragraphs:\n{paragraph_text}")

    history_text = format_history(history)
    if history_text:
        parts.append(f"Conversation history:\n{history_text}")

    return "\n\n".join(parts) if parts else "No supporting document."


def convert_split(raw_data: List[Dict[str, Any]], split: str, template_type: str, data_source: str) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for entry_idx, entry in enumerate(raw_data):
        questions = sorted(entry.get("questions", []), key=lambda q: q.get("order", 0))
        history: List[Dict[str, Any]] = []

        for turn_idx, q in enumerate(questions):
            question_text = str(q.get("question", "")).strip()
            if question_text and question_text[-1] not in {"?", "！", "。", "."}:
                question_text += "?"

            answers = normalize_answer(q.get("answer"))
            context = build_context(entry, history)
            prompt_text = make_prefix(question_text, context, template_type=template_type)

            data_item = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": answers},
                },
                "extra_info": {
                    "split": split,
                    "table_uid": (entry.get("table") or {}).get("uid"),
                    "question_uid": q.get("uid"),
                    "order": q.get("order", turn_idx),
                    "index": len(processed),
                    "entry_index": entry_idx,
                    "answer_type": q.get("answer_type"),
                    "answer_from": q.get("answer_from"),
                    "derivation": q.get("derivation", ""),
                    "scale": q.get("scale", ""),
                    "req_clari": q.get("req_clari", False),
                    "follow_up": q.get("follow_up", False),
                    "rel_paragraphs": q.get("rel_paragraphs", []),
                    "original_question": q.get("original_question", question_text),
                },
            }

            processed.append(data_item)

            history.append(
                {
                    "question": question_text,
                    "answer": answers,
                }
            )

    return processed


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=_default_data_dir())
    parser.add_argument("--local_dir", type=str, default="./data/pacific_fewshot",
                        help="输出目录（默认与原始数据分开）")
    parser.add_argument("--hdfs_dir", type=str, default=None)
    parser.add_argument("--template_type", type=str, default="fewshot",
                        help="提示模板类型: fewshot / base")
    args = parser.parse_args()

    data_source = "pacific"

    train_path = os.path.join(args.data_dir, "train.json")
    val_path = os.path.join(args.data_dir, "validation.json")
    test_path = os.path.join(args.data_dir, "test.json")

    print(f"Loading PACIFIC data from {args.data_dir}")
    print(f"Template type: {args.template_type}")
    train_raw = load_json(train_path)
    val_raw = load_json(val_path)
    test_raw = load_json(test_path)

    print("Converting splits...")
    train_processed = convert_split(train_raw, "train", args.template_type, data_source)
    val_processed = convert_split(val_raw, "validation", args.template_type, data_source)
    test_processed = convert_split(test_raw, "test", args.template_type, data_source)

    print(f"Train: {len(train_processed)}, Val: {len(val_processed)}, Test: {len(test_processed)}")

    # 打印一个样例 prompt 供检查
    if train_processed:
        sample = train_processed[0]["prompt"][0]["content"]
        print(f"\n{'='*60}\nSample prompt (first 800 chars):\n{'='*60}")
        print(sample[:800])
        print(f"{'='*60}\n")

    train_ds = datasets.Dataset.from_list(train_processed)
    val_ds = datasets.Dataset.from_list(val_processed)
    test_ds = datasets.Dataset.from_list(test_processed)

    os.makedirs(args.local_dir, exist_ok=True)

    train_ds.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(args.local_dir, "validation.parquet"))
    test_ds.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)

    print("Done.")


if __name__ == "__main__":
    main()
