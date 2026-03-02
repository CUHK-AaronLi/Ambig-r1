"""
Preprocess AmbigQA dataset to parquet format (Few-Shot Prompt version)

基于 ambignq.py，更新 make_prefix 加入 few-shot 示例，
演示 search / clarify / answer 三种 action 的正确格式和使用时机。
"""

import re
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse


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
            "or a specific era. I should clarify which one the user means.</think>\n"
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
    elif template_type == 'base':
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, you can choose one of three actions:
1. If you need more information, use <search> query </search> to search and get results between <information> and </information>.
2. If the question is ambiguous or unclear, use <clarify> your clarification question </clarify> to communicate directly with the user through the message you write in the content field. Ask clear and specific questions to gather the information needed to fulfill the user's request. Keep in mind that the user may not have all the necessary details, so you might need to both request additional user input and call internal tools step by step to reach the goal. Good clarification questions help narrow down the scope:
   Focus on asking the user to choose between specific options or provide missing details that are essential for answering their question accurately.
3. If you have enough information, provide the answer inside <answer> and </answer>.

You can search or clarify as many times as needed but only when necessary. Question: {question}\n"""
    else:
        raise NotImplementedError(f"Unsupported template_type: {template_type}")
    return prefix


def process_ambiguous_instances(dataset, max_examples=None):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/ambignq_fewshot')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='fewshot')

    args = parser.parse_args()

    data_source = 'ambignq'

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'ambig_qa')

    train_dataset = dataset['train']
    test_dataset = dataset['dev']

    print(f"Original train: {len(train_dataset)}, test: {len(test_dataset)}")
    print(f"Template type: {args.template_type}")

    train_processed = process_ambiguous_instances(train_dataset)
    test_processed = process_ambiguous_instances(test_dataset)

    minitest_limit = 50
    minitest_processed = process_ambiguous_instances(test_dataset, max_examples=minitest_limit)
    tinytest_limit = 500
    tinytest_processed = process_ambiguous_instances(test_dataset, max_examples=tinytest_limit)

    print(f"Processed train: {len(train_processed)}, test: {len(test_processed)}")
    print(f"minitest: {len(minitest_processed)}, tinytest: {len(tinytest_processed)}")

    # 统计歧义/非歧义比例
    ambig_count = sum(1 for x in train_processed if x.get('_is_ambiguous'))
    print(f"Train ambiguous: {ambig_count}/{len(train_processed)} ({ambig_count/len(train_processed)*100:.1f}%)")

    train_dataset = datasets.Dataset.from_list(train_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)
    minitest_dataset = datasets.Dataset.from_list(minitest_processed)
    tinytest_dataset = datasets.Dataset.from_list(tinytest_processed)

    def make_map_fn(split):
        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)

            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "ambiguous-question-answering",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'original_idx': example.get('_original_idx', idx),
                    'qa_pair_idx': example.get('_qa_pair_idx', 0),
                    '_is_ambiguous': example.get('_is_ambiguous', False),
                    'gold_question': example.get('gold_question', ''),
                    'original_question': example.get('question', ''),
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    minitest_dataset = minitest_dataset.map(function=make_map_fn('minitest'), with_indices=True)
    tinytest_dataset = tinytest_dataset.map(function=make_map_fn('tinytest'), with_indices=True)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    minitest_dataset.to_parquet(os.path.join(local_dir, 'minitest.parquet'))
    tinytest_dataset.to_parquet(os.path.join(local_dir, 'tinytest.parquet'))

    # 打印一个样例 prompt
    if len(train_dataset) > 0:
        sample = train_dataset[0]['prompt'][0]['content']
        print(f"\n{'='*60}\nSample prompt (first 600 chars):\n{'='*60}")
        print(sample[:600])
        print(f"{'='*60}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print("Done.")
