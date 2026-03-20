"""
预处理 PACIFIC 数据集为统一 parquet 格式
参考 `nq_rag.py` 的转换逻辑，构造包含表格、段落和对话历史的上下文，
并输出用于训练/评测的标准字段。
"""

import os
import json
import argparse
from typing import List, Dict, Any

import datasets

from verl.utils.hdfs_io import copy, makedirs


def _default_data_dir() -> str:
    """返回默认的原始 PACIFIC 数据目录（相对仓库根目录）。"""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../data/PACIFIC/data/pacific")
    )


def make_prefix(question: str, context: str, template_type: str) -> str:
    """
    构造模型提示前缀。
    """
    if template_type == "base":
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
    """将答案标准化为字符串列表。"""
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(a).strip() for a in ans if str(a).strip()]
    return [str(ans).strip()]


def format_table(table_obj: Dict[str, Any]) -> str:
    """将表格字段转换为可读文本。"""
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
    """将段落按 order 排序后拼接。"""
    if not paragraphs:
        return ""
    paragraphs = sorted(paragraphs, key=lambda x: x.get("order", 0))
    texts = [p.get("text", "").strip() for p in paragraphs if p.get("text")]
    return "\n".join(texts)


def format_history(history: List[Dict[str, Any]]) -> str:
    """格式化对话历史：[{question, answer}] -> 多行文本。"""
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
    """组合表格、段落和历史为上下文字符串。"""
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
    """
    将原始 split 数据转换为训练所需格式。
    每个表格包含多个问题，逐个拆成样本并带上历史。
    """
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
                    "data_source": data_source,
                    # Simulator context: table structure helps GPT-4o generate
                    # meaningful clarification responses
                    "clarify_context": context,
                },
            }

            processed.append(data_item)

            # 更新历史，供后续 turns 使用
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
    parser.add_argument("--data_dir", type=str, default=_default_data_dir(), help="原始 PACIFIC 数据所在目录，包含 train/validation/test.json")
    parser.add_argument("--local_dir", type=str, default="./data/pacific", help="输出 parquet 保存目录")
    parser.add_argument("--hdfs_dir", type=str, default=None, help="可选：HDFS 目标目录")
    parser.add_argument("--template_type", type=str, default="base", help="提示模板类型")
    args = parser.parse_args()

    data_source = "pacific"

    train_path = os.path.join(args.data_dir, "train.json")
    val_path = os.path.join(args.data_dir, "validation.json")
    test_path = os.path.join(args.data_dir, "test.json")

    print(f"Loading PACIFIC data from {args.data_dir}")
    train_raw = load_json(train_path)
    val_raw = load_json(val_path)
    test_raw = load_json(test_path)

    print("Converting splits...")
    train_processed = convert_split(train_raw, "train", args.template_type, data_source)
    val_processed = convert_split(val_raw, "validation", args.template_type, data_source)
    test_processed = convert_split(test_raw, "test", args.template_type, data_source)

    print(f"Train samples: {len(train_processed)}, Val samples: {len(val_processed)}, Test samples: {len(test_processed)}")

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
