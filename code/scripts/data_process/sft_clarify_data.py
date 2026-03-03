"""
Generate SFT training data from AmbigNQ with gold clarify->answer trajectories.

For ambiguous questions (multipleQAs): produces a gold first-turn trajectory
  <think>...</think>
  <clarify>disambiguating question</clarify>

For non-ambiguous questions: produces a direct answer
  <think>...</think>
  <answer>gold answer</answer>

NOTE: We only train on the FIRST TURN of model output. For ambiguous questions,
this means only the <think>+<clarify> part — NOT the <user_response> (which is
environment-injected) or the follow-up <think>+<answer>. This matches the RL
environment where the model generates up to </clarify>, then the environment
provides <user_response>, and only then the model continues.

Output: Parquet with 'prompt' (plain string) and 'response' (plain string)
columns for fsdp_sft_trainer.py SFTDataset.
Balance: ~50% ambiguous, ~50% non-ambiguous.
"""

import os
import random
import argparse
import datasets


def make_prefix(question):
    """System prompt — identical to ambignq_fewshot.py fewshot template.

    IMPORTANT: Must use the exact same Unicode characters (→, —) as
    ambignq_fewshot.py to avoid distribution mismatch between SFT and RL.
    """
    prefix = (
        "Answer the given question. You must reason inside <think> and </think> first.\n"
        "After reasoning, choose one of three actions:\n"
        "\n"
        "1. **Search**: If you need more information, use <search>query</search>.\n"
        "   Results will appear between <information> and </information>.\n"
        "2. **Clarify**: If the question is ambiguous, use <clarify>your question</clarify>.\n"
        "   The user's response will appear between <user_response> and </user_response>.\n"
        "3. **Answer**: If you are confident, use <answer>your answer</answer>.\n"
        "\n"
        "You MUST use the exact XML tags shown above. Do NOT omit closing tags.\n"
        "\n"
        "## Example 1: Ambiguous question \u2192 Clarify \u2192 Answer\n"
        "\n"
        "Question: When was the last time the Yankees won the World Series?\n"
        "\n"
        "<think>This question could refer to different things \u2014 their most recent win, "
        "or a specific era. I should clarify which one the user means.</think>\n"
        "<clarify>Are you asking about their most recent World Series win, "
        "or a win in a specific decade?</clarify>\n"
        "\n"
        "<user_response>Their most recent win.</user_response>\n"
        "\n"
        "<think>The user wants the most recent win. The Yankees last won in 2009.</think>\n"
        "<answer>2009</answer>\n"
        "\n"
        "## Example 2: Need information \u2192 Search \u2192 Answer\n"
        "\n"
        "Question: What is the population of Zurich?\n"
        "\n"
        "<think>I'm not sure about the exact number. Let me search for it.</think>\n"
        "<search>population of Zurich</search>\n"
        "\n"
        "<information>[1] Zurich has a population of approximately 434,008 (2022).</information>\n"
        "\n"
        "<think>The search result gives me the answer.</think>\n"
        "<answer>434,008</answer>\n"
        "\n"
        "## Example 3: Clear question \u2192 Answer directly\n"
        "\n"
        "Question: Who wrote Romeo and Juliet?\n"
        "\n"
        "<think>This is a well-known fact. No ambiguity, no search needed.</think>\n"
        "<answer>William Shakespeare</answer>\n"
        "\n"
        "---\n"
        "Now answer the following question. Use <clarify> if ambiguous, "
        "<search> if you need information, or <answer> when confident.\n"
        f"\nQuestion: {question}\n"
    )
    return prefix


def process_ambiguous_instances(dataset, max_examples=None):
    """Same logic as ambignq_fewshot.py — process AmbigNQ instances."""
    processed_instances = []

    for idx, example in enumerate(dataset):
        if (example.get('metadata') and
            example['metadata'].get('annotation') and
            len(example['metadata']['annotation']) > 0 and
            example['metadata']['annotation'][0].get('type') == 'multipleQAs'):

            qa_pairs = example['metadata']['annotation'][0].get('qaPairs', [])

            for qa_idx, qa_pair in enumerate(qa_pairs):
                new_instance = example.copy()
                new_instance['gold_question'] = qa_pair['question']
                new_instance['golden_answers'] = qa_pair['answer']
                new_instance['id'] = f"train_{idx}_{qa_idx}"
                new_instance['_original_idx'] = idx
                new_instance['_qa_pair_idx'] = qa_idx
                new_instance['_is_ambiguous'] = True
                processed_instances.append(new_instance)
        else:
            example['gold_question'] = example['question']
            example['id'] = f"train_{idx}_0"
            example['_original_idx'] = idx
            example['_qa_pair_idx'] = 0
            example['_is_ambiguous'] = False
            processed_instances.append(example)

    for i, instance in enumerate(processed_instances):
        instance['idx'] = i

    return processed_instances[:max_examples] if max_examples is not None else processed_instances


def make_clarify_question(original_question, gold_question):
    """Generate a clarifying question from the gold disambiguated question.

    The gold_question is the specific interpretation (e.g., "When did the
    Yankees most recently win?"), so we turn it into a clarifying question
    that would elicit this interpretation.
    """
    # If the gold question is very similar to original, make a generic clarify
    if gold_question.lower().strip('?').strip() == original_question.lower().strip('?').strip():
        return f"Could you be more specific about what you mean by \"{original_question}\"?"

    return f"Are you asking: {gold_question}"


def build_gold_response_ambiguous(original_question, gold_question, gold_answers):
    """Build a gold FIRST-TURN response for an ambiguous question.

    Only includes the model's first action: <think> + <clarify>.
    Does NOT include <user_response> (environment-injected) or the
    follow-up <think> + <answer> (second model turn).
    """
    clarify_q = make_clarify_question(original_question, gold_question)

    response = (
        f"<think>This question is ambiguous. It could have multiple interpretations. "
        f"I should clarify what the user means.</think>\n"
        f"<clarify>{clarify_q}</clarify>"
    )
    return response


def build_gold_response_direct(gold_answers):
    """Build a direct answer trajectory for a non-ambiguous question."""
    answer = gold_answers[0] if isinstance(gold_answers, list) else str(gold_answers)

    response = (
        f"<think>This question is clear and I can answer it directly.</think>\n"
        f"<answer>{answer}</answer>"
    )
    return response


def main():
    parser = argparse.ArgumentParser(description='Generate SFT clarify training data')
    parser.add_argument('--local_dir', default='./data/sft_clarify')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--max_train', type=int, default=None,
                        help='Max training examples (None = use all)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # Load AmbigNQ dataset (same source as ambignq_fewshot.py)
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'ambig_qa')

    train_dataset = dataset['train']
    test_dataset = dataset['dev']

    print(f"Original train: {len(train_dataset)}, test: {len(test_dataset)}")

    train_processed = process_ambiguous_instances(train_dataset)
    test_processed = process_ambiguous_instances(test_dataset)

    print(f"Processed train: {len(train_processed)}, test: {len(test_processed)}")

    # Build SFT examples
    sft_examples = []

    for instance in train_processed:
        question = instance['question'].strip()
        if question[-1] != '?':
            question += '?'

        prompt = make_prefix(question)

        golden_answers = instance.get('golden_answers', [''])
        if not golden_answers:
            continue

        if instance.get('_is_ambiguous', False):
            gold_question = instance.get('gold_question', question)
            response = build_gold_response_ambiguous(question, gold_question, golden_answers)
        else:
            response = build_gold_response_direct(golden_answers)

        sft_examples.append({
            'prompt': prompt,
            'response': response,
        })

    # Balance ambiguous vs non-ambiguous
    ambig_examples = [ex for ex in sft_examples if '<clarify>' in ex['response']]
    non_ambig_examples = [ex for ex in sft_examples if '<clarify>' not in ex['response']]

    print(f"Before balancing: ambiguous={len(ambig_examples)}, non-ambiguous={len(non_ambig_examples)}")

    # Downsample the majority class to achieve ~50/50 balance
    min_count = min(len(ambig_examples), len(non_ambig_examples))
    if len(ambig_examples) > min_count:
        random.shuffle(ambig_examples)
        ambig_examples = ambig_examples[:min_count]
    if len(non_ambig_examples) > min_count:
        random.shuffle(non_ambig_examples)
        non_ambig_examples = non_ambig_examples[:min_count]

    balanced_examples = ambig_examples + non_ambig_examples
    random.shuffle(balanced_examples)

    if args.max_train is not None:
        balanced_examples = balanced_examples[:args.max_train]

    print(f"After balancing: {len(balanced_examples)} total examples")
    print(f"  Ambiguous: {sum(1 for ex in balanced_examples if '<clarify>' in ex['response'])}")
    print(f"  Non-ambiguous: {sum(1 for ex in balanced_examples if '<clarify>' not in ex['response'])}")

    # Build validation set (smaller, unbalanced is fine)
    val_examples = []
    for instance in test_processed[:500]:
        question = instance['question'].strip()
        if question[-1] != '?':
            question += '?'

        prompt = make_prefix(question)

        golden_answers = instance.get('golden_answers', [''])
        if not golden_answers:
            continue

        if instance.get('_is_ambiguous', False):
            gold_question = instance.get('gold_question', question)
            response = build_gold_response_ambiguous(question, gold_question, golden_answers)
        else:
            response = build_gold_response_direct(golden_answers)

        val_examples.append({
            'prompt': prompt,
            'response': response,
        })

    print(f"Validation examples: {len(val_examples)}")

    # Save as parquet
    os.makedirs(args.local_dir, exist_ok=True)

    train_ds = datasets.Dataset.from_list(balanced_examples)
    val_ds = datasets.Dataset.from_list(val_examples)

    train_ds.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    val_ds.to_parquet(os.path.join(args.local_dir, 'val.parquet'))

    # Print a sample
    if len(balanced_examples) > 0:
        sample = balanced_examples[0]
        print(f"\n{'='*60}\nSample prompt (first 300 chars):\n{'='*60}")
        print(sample['prompt'][:300])
        print(f"\n{'='*60}\nSample response:\n{'='*60}")
        print(sample['response'])
        print(f"{'='*60}")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)

    print("Done.")


if __name__ == '__main__':
    main()
