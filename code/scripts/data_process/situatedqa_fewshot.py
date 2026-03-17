"""
Preprocess SituatedQA dataset to parquet format for IPO training.

SituatedQA contains questions whose answers depend on temporal or geographical context.
These are naturally ambiguous — asking "when did X happen" may have different answers
depending on which time period the user cares about.

We mark context-dependent questions as ambiguous and collect all valid answers.
"""

import os
import json
import datasets
import argparse
from collections import defaultdict


def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'fewshot':
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
            "## Example 1: Ambiguous question → Clarify → Answer\n"
            "\n"
            "Question: When was the last time the Yankees won the World Series?\n"
            "\n"
            "<think>This question could refer to different things — their most recent win, "
            "or a win in a specific decade. I should clarify which one the user means.</think>\n"
            "<clarify>Are you asking about their most recent World Series win, "
            "or a win in a specific decade?</clarify>\n"
            "\n"
            "<user_response>Their most recent win.</user_response>\n"
            "\n"
            "<think>The user wants the most recent win. The Yankees last won in 2009.</think>\n"
            "<answer>2009</answer>\n"
            "\n"
            "## Example 2: Need information → Search → Answer\n"
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
            "## Example 3: Clear question → Answer directly\n"
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
    else:
        raise NotImplementedError(f"Unsupported template_type: {template_type}")
    return prefix


def load_situatedqa():
    """Load SituatedQA from HuggingFace."""
    try:
        ds = datasets.load_dataset('siyue/SituatedQA')
        return ds
    except Exception:
        pass

    # Fallback: try GitHub raw files
    print("HuggingFace load failed, trying GitHub...")
    import urllib.request
    base_url = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/main/data"
    data = {}
    for split in ['train', 'dev', 'test']:
        url = f"{base_url}/{split}.jsonl"
        print(f"Downloading {url}...")
        response = urllib.request.urlopen(url)
        lines = response.read().decode('utf-8').strip().split('\n')
        data[split] = [json.loads(line) for line in lines]
    return data


def process_situatedqa(data, split_name, max_examples=None):
    """Process SituatedQA instances.

    Context-dependent questions (those with multiple answers for different contexts)
    are marked as ambiguous. All valid answers are collected as ground truth.
    """
    processed = []

    if isinstance(data, list):
        items = data
    else:
        items = list(data)

    for idx, item in enumerate(items):
        question = item.get('question', '')
        if not question:
            continue

        # Collect all valid answers across contexts
        all_answers = set()

        # SituatedQA format varies - handle different structures
        if 'answers' in item:
            # List of answer objects with context
            for ans_obj in item['answers']:
                if isinstance(ans_obj, dict):
                    ans = ans_obj.get('answer', ans_obj.get('value', ''))
                    if ans:
                        all_answers.add(str(ans))
                elif isinstance(ans_obj, str):
                    all_answers.add(ans_obj)
        elif 'answer' in item:
            ans = item['answer']
            if isinstance(ans, list):
                all_answers.update([str(a) for a in ans])
            else:
                all_answers.add(str(ans))

        # Determine ambiguity
        is_context_dependent = item.get('context_dependent', False)
        has_temporal = item.get('has_temporal_answer', False)
        has_geo = item.get('has_geo_answer', False)
        is_ambiguous = is_context_dependent or has_temporal or has_geo or len(all_answers) > 1

        # Get ambiguity type
        ambiguity_type = 'none'
        if has_temporal:
            ambiguity_type = 'temporal'
        elif has_geo:
            ambiguity_type = 'geographical'
        elif is_context_dependent:
            ambiguity_type = 'context_dependent'

        if not all_answers:
            all_answers = {'unknown'}

        processed.append({
            'question': question,
            'golden_answers': list(all_answers),
            '_is_ambiguous': is_ambiguous,
            'ambiguity_type': ambiguity_type,
            'source_dataset': item.get('source', 'situatedqa'),
            'original_id': item.get('id', f'{split_name}_{idx}'),
            'idx': idx,
        })

    if max_examples:
        processed = processed[:max_examples]

    return processed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/situatedqa_fewshot')
    parser.add_argument('--template_type', type=str, default='fewshot')
    args = parser.parse_args()

    data_source = 'situatedqa'

    print("Loading SituatedQA...")
    raw_data = load_situatedqa()

    if isinstance(raw_data, dict) and 'train' in raw_data:
        if hasattr(raw_data['train'], '__len__'):
            train_raw = raw_data['train']
            test_raw = raw_data.get('dev', raw_data.get('validation', raw_data.get('test', [])))
        else:
            train_raw = list(raw_data['train'])
            test_raw = list(raw_data.get('dev', raw_data.get('validation', raw_data.get('test', []))))
    else:
        # Single split, do manual split
        all_data = list(raw_data) if not isinstance(raw_data, list) else raw_data
        split_point = int(len(all_data) * 0.9)
        train_raw = all_data[:split_point]
        test_raw = all_data[split_point:]

    print(f"Raw train: {len(train_raw)}, test: {len(test_raw)}")

    train_processed = process_situatedqa(train_raw, 'train')
    test_processed = process_situatedqa(test_raw, 'test')
    tinytest_processed = process_situatedqa(test_raw, 'tinytest', max_examples=500)

    print(f"Processed train: {len(train_processed)}, test: {len(test_processed)}, tinytest: {len(tinytest_processed)}")

    ambig_count = sum(1 for x in train_processed if x['_is_ambiguous'])
    print(f"Train ambiguous: {ambig_count}/{len(train_processed)} ({ambig_count/len(train_processed)*100:.1f}%)")

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
                    'ambiguity_type': example['ambiguity_type'],
                    'original_question': example['question'],
                    'gold_question': example['question'],
                    'source_dataset': example['source_dataset'],
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
        print(f"\n{'='*60}\nSample prompt (first 600 chars):\n{'='*60}")
        print(sample[:600])
        print(f"\nAnswers: {train_dataset[0]['reward_model']['ground_truth']['target']}")
        print(f"Ambiguous: {train_dataset[0]['extra_info']['_is_ambiguous']}")
        print(f"{'='*60}")

    print("Done.")
