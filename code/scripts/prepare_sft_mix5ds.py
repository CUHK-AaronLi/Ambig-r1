#!/usr/bin/env python3
"""
Generate SFT training data from mixed 5 datasets (mix5ds).
Each example: prompt (system + context + question) + response (<clarify> or <answer>)

Dataset strategy:
- PACIFIC: Clarify examples (parameter ambiguity in tables)
- SHARC: Clarify examples (rule-following questions)
- AbgCoQA: Mix of clarify (ambiguous) and answer (non-ambiguous)
- AmbigNQ: Mix of clarify (ambiguous) and answer (non-ambiguous)
- SituatedQA: Mix of clarify (temporal/geographical) and answer

Output: parquet with 'prompt' and 'response' columns.
"""

import os
import random
import argparse
import pandas as pd


# ===== System prompt (same format as ambignq_fewshot.py) =====
SYSTEM_PROMPT = """Answer the given question. You must reason inside <think> and </think> first.
After reasoning, choose one of three actions:

1. **Search**: If you need more information, use <search>query</search>.
   Results will appear between <information> and </information>.
2. **Clarify**: If the question is ambiguous, use <clarify>your question</clarify>.
   The user's response will appear between <user_response> and </user_response>.
3. **Answer**: If you are confident, use <answer>your answer</answer>.

You MUST use the exact XML tags shown above. Do NOT omit closing tags.

## Example 1: Ambiguous question → Clarify → Answer

Question: When was the last time the Yankees won the World Series?

<think>This question could refer to different things — their most recent win, or a specific era. I should clarify which one the user means.</think>
<clarify>Are you asking about their most recent World Series win, or a win in a specific decade?</clarify>

## Example 2: Clear question → Answer directly

Question: Who wrote Romeo and Juliet?

<think>This is a well-known fact. No ambiguity, no search needed.</think>
<answer>William Shakespeare</answer>

---
Now answer the following question. Use <clarify> if ambiguous, <search> if you need information, or <answer> when confident.
"""


def make_prompt(question, context=None):
    """Build SFT prompt from question and optional context."""
    if context:
        return SYSTEM_PROMPT + f"\n{context}\n\nQuestion: {question}\n"
    return SYSTEM_PROMPT + f"\nQuestion: {question}\n"


def make_clarify_response(original_question, gold_question):
    """Build a clarification response."""
    # Extract the key disambiguation from gold_question
    gold_lower = gold_question.lower().strip('?')
    orig_lower = original_question.lower().strip('?')

    # If gold is same as original, make generic clarification
    if gold_lower == orig_lower or not gold_question.strip('?'):
        return (
            f"<think>The question could have multiple interpretations. "
            f"I should clarify what the user specifically means.</think>\n"
            f'<clarify>Could you be more specific about what you mean by "{original_question}"?</clarify>'
        )

    # Extract key difference: usually the parameter/value in gold_question
    # e.g., "What is revenue in 2019?" vs "What is revenue in 2020?"
    return (
        f"<think>The question is ambiguous and could refer to different specific cases. "
        f"The disambiguated question specifies: {gold_question}. I should clarify.</think>\n"
        f'<clarify>Are you asking: {gold_question}?</clarify>'
    )


def make_answer_response(gold_answer):
    """Build a direct answer response."""
    answer = gold_answer if isinstance(gold_answer, str) else str(gold_answer)
    return (
        f"<think>The question is clear and I can answer directly.</think>\n"
        f"<answer>{answer}</answer>"
    )


def extract_gold_answer(reward_model):
    """Extract gold answer from reward_model field."""
    gt = reward_model.get('ground_truth', {}) if isinstance(reward_model, dict) else {}
    target = gt.get('target', ['N/A'])
    if hasattr(target, 'tolist'):
        target = target.tolist()
    return target[0] if target and target[0] else 'N/A'


def process_pacific(df, n_samples=400, seed=42):
    """PACIFIC: always clarify (parameter ambiguity in tables)."""
    random.seed(seed)
    pac = df[df['data_source'] == 'pacific'].copy()

    examples = []
    for _, row in pac.iterrows():
        ei = row['extra_info']
        gold_q = ei.get('gold_question', '') or ''
        original_q = ei.get('original_question', '') or gold_q
        gold_a = extract_gold_answer(row['reward_model'])
        table_ctx = ei.get('clarify_context', '') or ''

        if not gold_a or gold_a == 'N/A':
            continue

        # Build context from table
        if table_ctx:
            context = f"Context:\n{table_ctx[:2000]}\n"
        else:
            context = None

        prompt = make_prompt(original_q, context)
        response = make_clarify_response(original_q, gold_q)

        examples.append({'prompt': prompt, 'response': response, 'source': 'pacific'})
        examples.append({'prompt': make_prompt(gold_q, context),
                       'response': make_answer_response(gold_a),
                       'source': 'pacific_gold'})

    random.shuffle(examples)
    return examples[:n_samples]


def process_sharc(df, n_samples=400, seed=42):
    """SHARC: always clarify (rule-following follow-up questions)."""
    random.seed(seed)
    sh = df[df['data_source'] == 'sharc'].copy()

    examples = []
    for _, row in sh.iterrows():
        ei = row['extra_info']
        gold_q = ei.get('gold_question', '') or ''
        original_q = ei.get('original_question', '') or gold_q
        gold_a = ei.get('original_answer', '') or 'N/A'
        scenario = ei.get('scenario', '') or ''

        if not gold_a or gold_a == 'N/A':
            continue

        # Build context from scenario
        if scenario:
            context = f"Rule Scenario:\n{scenario[:2000]}\n"
        else:
            context = None

        prompt = make_prompt(original_q, context)
        response = make_clarify_response(original_q, gold_q)

        examples.append({'prompt': prompt, 'response': response, 'source': 'sharc'})
        examples.append({'prompt': make_prompt(gold_q, context),
                       'response': make_answer_response(gold_a),
                       'source': 'sharc_gold'})

    random.shuffle(examples)
    return examples[:n_samples]


def process_abgcoqa(df, n_samples=500, seed=42):
    """AbgCoQA: ambiguous → clarify, non-ambiguous → answer."""
    random.seed(seed)
    abg = df[df['data_source'] == 'abgcoqa'].copy()

    examples = []
    for _, row in abg.iterrows():
        ei = row['extra_info']
        is_ambig = ei.get('ambiguity') == 'ambiguous'
        gold_q = ei.get('gold_question', '') or ''
        original_q = ei.get('original_question', '') or gold_q
        target_turn = ei.get('target_turn', {})
        if target_turn is None:
            target_turn = {}
        if hasattr(target_turn, 'tolist'):
            target_turn = target_turn.tolist()
        if hasattr(target_turn, 'item'):
            target_turn = target_turn.item()
        gold_a = target_turn.get('answer', 'N/A') if isinstance(target_turn, dict) else 'N/A'
        history = ei.get('history_turns', [])
        if history is None:
            history = []
        if hasattr(history, 'tolist'):
            history = history.tolist()
        ctx = ei.get('user_simulator_context', '') or ''

        if not gold_a or gold_a == 'N/A' or not gold_q:
            continue

        # Build conversation history context
        if history and isinstance(history, list):
            ctx_lines = []
            for turn in history[:6]:  # Last 6 turns
                if isinstance(turn, dict):
                    q = turn.get('question', '')
                    a = turn.get('answer', '')
                    if q or a:
                        ctx_lines.append(f"Q: {q}\nA: {a}")
            if ctx_lines:
                history_ctx = '\n'.join(ctx_lines)
                context = f"Conversation History:\n{history_ctx}\n"
            else:
                context = None
        else:
            context = None

        if is_ambig:
            # Ambiguous → clarify + gold → answer
            prompt = make_prompt(original_q, context)
            response = make_clarify_response(original_q, gold_q)
            examples.append({'prompt': prompt, 'response': response, 'source': 'abgcoqa_ambig'})
            examples.append({'prompt': make_prompt(gold_q, context),
                           'response': make_answer_response(gold_a),
                           'source': 'abgcoqa_gold'})
        else:
            # Non-ambiguous → direct answer
            prompt = make_prompt(original_q, context)
            response = make_answer_response(gold_a)
            examples.append({'prompt': prompt, 'response': response, 'source': 'abgcoqa_direct'})

    random.shuffle(examples)
    return examples[:n_samples]


def process_ambignq(df, n_samples=500, seed=42):
    """AmbigNQ: ambiguous → clarify, non-ambiguous → answer."""
    random.seed(seed)
    amb = df[df['data_source'] == 'ambignq'].copy()

    examples = []
    for _, row in amb.iterrows():
        is_ambig = row.get('_is_ambiguous', False)
        gold_q = row.get('gold_question', '') or row.get('question', '') or ''
        original_q = row.get('question', '') or gold_q
        gold_answers = row.get('golden_answers', []) or []
        gold_a = gold_answers[0] if gold_answers else 'N/A'

        if not gold_a or gold_a == 'N/A' or not gold_q:
            continue

        if is_ambig:
            prompt = make_prompt(original_q)
            response = make_clarify_response(original_q, gold_q)
            examples.append({'prompt': prompt, 'response': response, 'source': 'ambignq_ambig'})
            examples.append({'prompt': make_prompt(gold_q),
                           'response': make_answer_response(gold_a),
                           'source': 'ambignq_gold'})
        else:
            prompt = make_prompt(original_q)
            response = make_answer_response(gold_a)
            examples.append({'prompt': prompt, 'response': response, 'source': 'ambignq_direct'})

    random.shuffle(examples)
    return examples[:n_samples]


def process_situatedqa(df, n_samples=500, seed=42):
    """SituatedQA: temporal/geographical → clarify + answer."""
    random.seed(seed)
    sit = df[df['data_source'] == 'situatedqa'].copy()

    examples = []
    for _, row in sit.iterrows():
        ei = row['extra_info']
        gold_q = ei.get('gold_question', '') or ''
        original_q = ei.get('original_question', '') or gold_q
        gold_answers = row.get('golden_answers', []) or []
        gold_a = gold_answers[0] if gold_answers else row.get('golden_answers', 'N/A')
        if hasattr(gold_a, 'tolist'):
            gold_a = gold_a[0] if gold_a else 'N/A'
        ambig_type = ei.get('ambiguity_type', '') or ''

        if not gold_a or gold_a == 'N/A' or not gold_q:
            continue

        # Always clarify: SituatedQA has temporal/spatial ambiguity
        prompt = make_prompt(original_q)
        response = make_clarify_response(original_q, gold_q)
        examples.append({'prompt': prompt, 'response': response, 'source': f'situatedqa_{ambig_type}'})
        examples.append({'prompt': make_prompt(gold_q),
                       'response': make_answer_response(gold_a),
                       'source': f'situatedqa_{ambig_type}_gold'})

    random.shuffle(examples)
    return examples[:n_samples]


def main():
    parser = argparse.ArgumentParser(description='Generate mixed 5-dataset SFT data')
    parser.add_argument('--input_parquet',
                       default='scripts/data_process/data/mix_5ds/train.parquet',
                       help='Path to mix_5ds train.parquet')
    parser.add_argument('--output_dir',
                       default='scripts/data_process/data/sft_mix5ds',
                       help='Output directory')
    parser.add_argument('--n_per_dataset', type=int, default=400,
                        help='Max examples per dataset (before balancing)')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    print(f"Loading from {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    print(f"Total rows: {len(df)}")
    print(f"Sources: {df['data_source'].value_counts().to_dict()}")

    # Process each dataset
    all_examples = []

    print("\n--- Processing PACIFIC ---")
    pac_ex = process_pacific(df, n_samples=args.n_per_dataset, seed=args.seed)
    print(f"  Generated {len(pac_ex)} examples")
    all_examples.extend(pac_ex)

    print("\n--- Processing SHARC ---")
    sh_ex = process_sharc(df, n_samples=args.n_per_dataset, seed=args.seed)
    print(f"  Generated {len(sh_ex)} examples")
    all_examples.extend(sh_ex)

    print("\n--- Processing AbgCoQA ---")
    abg_ex = process_abgcoqa(df, n_samples=args.n_per_dataset, seed=args.seed)
    print(f"  Generated {len(abg_ex)} examples")
    all_examples.extend(abg_ex)

    print("\n--- Processing AmbigNQ ---")
    amb_ex = process_ambignq(df, n_samples=args.n_per_dataset, seed=args.seed)
    print(f"  Generated {len(amb_ex)} examples")
    all_examples.extend(amb_ex)

    print("\n--- Processing SituatedQA ---")
    sit_ex = process_situatedqa(df, n_samples=args.n_per_dataset, seed=args.seed)
    print(f"  Generated {len(sit_ex)} examples")
    all_examples.extend(sit_ex)

    # Balance: ~50% clarify, ~50% answer
    clarify_ex = [e for e in all_examples if '<clarify>' in e['response']]
    answer_ex = [e for e in all_examples if '<clarify>' not in e['response']]

    print(f"\nBefore balancing: clarify={len(clarify_ex)}, answer={len(answer_ex)}")

    # Downsample majority to ~50/50
    min_count = min(len(clarify_ex), len(answer_ex))
    if len(clarify_ex) > min_count:
        random.shuffle(clarify_ex)
        clarify_ex = clarify_ex[:min_count]
    if len(answer_ex) > min_count:
        random.shuffle(answer_ex)
        answer_ex = answer_ex[:min_count]

    balanced = clarify_ex + answer_ex
    random.shuffle(balanced)
    print(f"After balancing: {len(balanced)} examples ({len(clarify_ex)} clarify, {len(answer_ex)} answer)")

    # Source breakdown
    source_counts = {}
    for e in balanced:
        src = e.pop('source')
        source_counts[src] = source_counts.get(src, 0) + 1
    print(f"Source breakdown: {source_counts}")

    # Split
    n_val = max(1, int(len(balanced) * args.val_ratio))
    val_examples = balanced[:n_val]
    train_examples = balanced[n_val:]
    print(f"\nTrain: {len(train_examples)}, Val: {len(val_examples)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, 'train.parquet')
    val_path = os.path.join(args.output_dir, 'val.parquet')

    pd.DataFrame(train_examples)[['prompt', 'response']].to_parquet(train_path, index=False)
    pd.DataFrame(val_examples)[['prompt', 'response']].to_parquet(val_path, index=False)

    print(f"\nSaved train to {train_path}")
    print(f"Saved val to {val_path}")

    # Sample
    if train_examples:
        sample = train_examples[0]
        print(f"\n{'='*60}\nSample (source: {sample.get('source', 'N/A')}):\n{'='*60}")
        print(f"Prompt (first 300 chars):\n{sample['prompt'][:300]}")
        print(f"\nResponse:\n{sample['response']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
