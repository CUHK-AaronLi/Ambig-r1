"""
Preprocess ShARC dataset to parquet format for IPO training.

ShARC (Shaping Answers with Rules through Conversation) is a conversational
machine reading dataset where the model must:
- Read a rule snippet (e.g., government policy)
- Decide whether to answer (Yes/No/Irrelevant) or ask a follow-up question (Inquire)

The "Inquire" label maps directly to our "clarify" action.
The "answer" labels (Yes/No/Irrelevant) map to our "answer" action.
The rule snippet lookup maps to our "search" action.

This creates a natural {clarify, search, answer} decision space.
"""

import os
import json
import datasets
import argparse


def make_prefix(dp, template_type):
    question = dp['question']
    scenario = dp.get('scenario', '')
    history = dp.get('history', [])

    # Build context from scenario and history
    context_parts = []
    if scenario:
        context_parts.append(f"User's situation: {scenario}")
    if history:
        for turn in history:
            if isinstance(turn, dict):
                q = turn.get('follow_up_question', turn.get('question', ''))
                a = turn.get('follow_up_answer', turn.get('answer', ''))
                if q and a:
                    context_parts.append(f"Previous Q: {q}")
                    context_parts.append(f"Previous A: {a}")
            elif isinstance(turn, list) and len(turn) == 2:
                context_parts.append(f"Previous Q: {turn[0]}")
                context_parts.append(f"Previous A: {turn[1]}")

    context_str = "\n".join(context_parts) if context_parts else ""

    if template_type == 'fewshot':
        prefix = (
            "Answer the given question based on rules and regulations. "
            "You must reason inside <think> and </think> first.\n"
            "After reasoning, choose one of three actions:\n"
            "\n"
            "1. **Search**: If you need to look up the relevant rule or regulation, "
            "use <search>query</search>.\n"
            "   Results will appear between <information> and </information>.\n"
            "2. **Clarify**: If you need more information from the user to determine "
            "the answer, use <clarify>your question</clarify>.\n"
            "   The user's response will appear between <user_response> and </user_response>.\n"
            "3. **Answer**: If you can determine the answer, use <answer>your answer</answer>.\n"
            "   Answers should be Yes, No, or a specific response.\n"
            "\n"
            "You MUST use the exact XML tags shown above. Do NOT omit closing tags.\n"
            "\n"
            "## Example 1: Need more user info → Clarify → Answer\n"
            "\n"
            "Question: Can I get a refund on my train ticket?\n"
            "User's situation: I bought a ticket last week.\n"
            "\n"
            "<think>The refund policy depends on when the trip is and whether "
            "the ticket is flexible. I need to ask for more details.</think>\n"
            "<clarify>When is your scheduled trip, and is your ticket "
            "an advance or flexible ticket?</clarify>\n"
            "\n"
            "<user_response>My trip is tomorrow and it's an advance ticket.</user_response>\n"
            "\n"
            "<think>Advance tickets for trips within 24 hours are usually non-refundable.</think>\n"
            "<answer>No</answer>\n"
            "\n"
            "## Example 2: Need to look up rules → Search → Answer\n"
            "\n"
            "Question: Am I eligible for the senior discount?\n"
            "User's situation: I am 63 years old.\n"
            "\n"
            "<think>I need to check the age requirement for the senior discount.</think>\n"
            "<search>senior discount age requirement eligibility</search>\n"
            "\n"
            "<information>[1] The senior discount is available to customers aged 65 and above.</information>\n"
            "\n"
            "<think>The user is 63, which is below the 65 threshold.</think>\n"
            "<answer>No</answer>\n"
            "\n"
            "## Example 3: Clear answer from context → Answer directly\n"
            "\n"
            "Question: Is this service available on weekends?\n"
            "User's situation: I want to use the service on Saturday.\n"
            "\n"
            "<think>The user wants to use it on Saturday. I know this service operates Monday to Saturday.</think>\n"
            "<answer>Yes</answer>\n"
            "\n"
            "---\n"
            "Now answer the following question. Use <clarify> if you need more user info, "
            "<search> if you need to look up rules, or <answer> when you can determine the answer.\n"
        )

        if context_str:
            prefix += f"\n{context_str}\n"
        prefix += f"\nQuestion: {question}\n"
    else:
        raise NotImplementedError(f"Unsupported template_type: {template_type}")

    return prefix


def process_sharc(dataset, split_name, max_examples=None):
    """Process ShARC instances.

    Label mapping:
    - "Inquire" -> ambiguous (needs clarification)
    - "Yes"/"No"/"Irrelevant" -> not ambiguous (can answer directly)
    """
    processed = []

    items = list(dataset) if not isinstance(dataset, list) else dataset

    for idx, item in enumerate(items):
        # Extract fields
        question = item.get('question', item.get('utterance', ''))
        if not question:
            continue

        snippet = item.get('snippet', item.get('clause', ''))
        scenario = item.get('scenario', '')
        history = item.get('history', item.get('evidence', []))
        answer = item.get('answer', '')
        follow_up = item.get('follow_up_question', '')

        # Determine if this is a clarification case
        # ShARC: answer is Yes/No/Irrelevant for clear cases,
        # or a follow-up question string for Inquire cases
        is_inquire = answer.lower() not in ('yes', 'no', 'irrelevant')

        # Build ground truth answers
        if is_inquire:
            # For Inquire cases, the answer field IS the follow-up question
            # The eventual answer is Yes/No, so we use both as valid answers
            follow_up = answer  # The follow-up question is stored in answer field
            golden_answers = ['Yes', 'No']
            is_ambiguous = True
        else:
            follow_up = ''
            golden_answers = [answer]
            is_ambiguous = False

        processed.append({
            'question': question,
            'golden_answers': golden_answers,
            '_is_ambiguous': is_ambiguous,
            'snippet': snippet,
            'scenario': scenario,
            'history': history,
            'original_answer': answer,
            'follow_up_question': follow_up,
            'idx': idx,
        })

    if max_examples:
        processed = processed[:max_examples]

    return processed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/sharc_fewshot')
    parser.add_argument('--template_type', type=str, default='fewshot')
    args = parser.parse_args()

    data_source = 'sharc'

    print("Loading ShARC from HuggingFace...")
    try:
        raw_data = datasets.load_dataset('UCLNLP/sharc')
    except Exception:
        print("Standard load failed, trying parquet revision...")
        raw_data = datasets.load_dataset('UCLNLP/sharc', revision='refs/convert/parquet')

    train_raw = raw_data['train']
    val_raw = raw_data.get('validation', raw_data.get('dev', []))
    test_raw = raw_data.get('test', [])

    print(f"Raw train: {len(train_raw)}, val: {len(val_raw)}, test: {len(test_raw)}")

    train_processed = process_sharc(train_raw, 'train')
    test_processed = process_sharc(val_raw if len(val_raw) > 0 else test_raw, 'test')
    tinytest_processed = process_sharc(
        val_raw if len(val_raw) > 0 else test_raw, 'tinytest', max_examples=500
    )

    print(f"Processed train: {len(train_processed)}, test: {len(test_processed)}, tinytest: {len(tinytest_processed)}")

    ambig_count = sum(1 for x in train_processed if x['_is_ambiguous'])
    print(f"Train Inquire (ambiguous): {ambig_count}/{len(train_processed)} ({ambig_count/len(train_processed)*100:.1f}%)")

    # Convert to HF datasets
    train_dataset = datasets.Dataset.from_list(train_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)
    tinytest_dataset = datasets.Dataset.from_list(tinytest_processed)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['question'].strip()
            if question and question[-1] != '?':
                question += '?'
            example['question'] = question

            prompt_text = make_prefix(example, template_type=args.template_type)

            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt_text,
                }],
                "ability": "ambiguous-question-answering",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    '_is_ambiguous': example['_is_ambiguous'],
                    'original_question': example['question'],
                    'gold_question': example['question'],
                    'snippet': example['snippet'],
                    'scenario': example['scenario'],
                    'original_answer': example['original_answer'],
                    'follow_up_question': example['follow_up_question'],
                    'data_source': data_source,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    tinytest_dataset = tinytest_dataset.map(function=make_map_fn('tinytest'), with_indices=True)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    tinytest_dataset.to_parquet(os.path.join(local_dir, 'tinytest.parquet'))

    if len(train_dataset) > 0:
        sample = train_dataset[0]['prompt'][0]['content']
        print(f"\n{'='*60}\nSample prompt (first 800 chars):\n{'='*60}")
        print(sample[:800])
        print(f"\nAnswers: {train_dataset[0]['reward_model']['ground_truth']['target']}")
        print(f"Ambiguous: {train_dataset[0]['extra_info']['_is_ambiguous']}")
        print(f"Original answer: {train_dataset[0]['extra_info']['original_answer']}")
        print(f"{'='*60}")

    print("Done.")
