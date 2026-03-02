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
Preprocess the AmbigQA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """基础模板：支持多动作（搜索、澄清、回答）"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, you can choose one of three actions:
1. If you need more information, use <search> query </search> to search and get results between <information> and </information>.
2. If the question is ambiguous or unclear, use <clarify> your clarification question </clarify> to communicate directly with the user through the message you write in the content field. Ask clear and specific questions to gather the information needed to fulfill the user's request. Keep in mind that the user may not have all the necessary details, so you might need to both request additional user input and call internal tools step by step to reach the goal. Good clarification questions help narrow down the scope:
   Focus on asking the user to choose between specific options or provide missing details that are essential for answering their question accurately.
3. If you have enough information, provide the answer inside <answer> and </answer>.

You can search or clarify as many times as needed but only when necessary. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


def process_ambiguous_instances(dataset, max_examples=None):
    """
    处理ambiguous实例，将multipleQAs类型的实例拆分成多个独立的实例
    """
    processed_instances = []
    
    for idx, example in enumerate(dataset):
        # 检查是否为ambiguous实例
        if (example.get('metadata') and 
            example['metadata'].get('annotation') and 
            len(example['metadata']['annotation']) > 0 and
            example['metadata']['annotation'][0].get('type') == 'multipleQAs'):
            
            # 这是一个ambiguous实例，需要拆分
            qa_pairs = example['metadata']['annotation'][0].get('qaPairs', [])
            
            for qa_idx, qa_pair in enumerate(qa_pairs):
                # 创建新的实例
                new_instance = example.copy()
                # 保持原来的question不变
                # new_instance['question'] = qa_pair['question']  # 删除这行
                # 新增gold_question字段，使用qaPair中的具体问题
                new_instance['gold_question'] = qa_pair['question']
                new_instance['golden_answers'] = qa_pair['answer']
                # 生成统一的ID格式
                new_instance['id'] = f"train_{idx}_{qa_idx}"
                new_instance['_original_idx'] = idx
                new_instance['_qa_pair_idx'] = qa_idx
                new_instance['_is_ambiguous'] = True
                processed_instances.append(new_instance)
        else:
            # 非ambiguous实例，保持原样，gold_question就是原question
            example['gold_question'] = example['question']
            # 生成统一的ID格式
            example['id'] = f"train_{idx}_0"
            example['_original_idx'] = idx
            example['_qa_pair_idx'] = 0
            example['_is_ambiguous'] = False
            processed_instances.append(example)
    
    # 为所有实例添加连续的idx字段
    for i, instance in enumerate(processed_instances):
        instance['idx'] = i
    
    return processed_instances[:max_examples] if max_examples is not None else processed_instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/ambignq')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'ambignq'

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'ambig_qa')

    train_dataset = dataset['train']
    test_dataset = dataset['dev']
    
    print("Original train dataset size:", len(train_dataset))
    print("Original test dataset size:", len(test_dataset))
    
    # 处理ambiguous实例
    print("Processing ambiguous instances...")
    train_processed = process_ambiguous_instances(train_dataset)
    test_processed = process_ambiguous_instances(test_dataset)
    
    # 生成minitest数据集（前50条）
    minitest_limit = 50
    minitest_processed = process_ambiguous_instances(test_dataset, max_examples=minitest_limit)
    
    # 生成tinytest数据集（前500条）
    tinytest_limit = 500
    tinytest_processed = process_ambiguous_instances(test_dataset, max_examples=tinytest_limit)
    
    print("Processed train dataset size:", len(train_processed))
    print("Processed test dataset size:", len(test_processed))
    print(f"Processed minitest dataset size: {len(minitest_processed)} (limited to {minitest_limit} examples)")
    print(f"Processed tinytest dataset size: {len(tinytest_processed)} (limited to {tinytest_limit} examples)")
    
    # 显示一些示例
    print("\nExample of processed instance:")
    print(train_processed[0])
    
    # 显示前十个实例的id、original_idx、qa_pair_idx信息
    print("\n" + "="*80)
    print("前十个实例的详细信息:")
    print("="*80)
    print(f"{'Idx':<6} {'ID':<15} {'Original_Idx':<15} {'QA_Pair_Idx':<12} {'Is_Ambiguous':<12}")
    print("-" * 80)
    
    for i in range(min(10, len(train_processed))):
        instance = train_processed[i]
        print(f"{instance.get('idx', 'N/A'):<6} {instance.get('id', 'N/A'):<15} {instance.get('_original_idx', 'N/A'):<15} {instance.get('_qa_pair_idx', 'N/A'):<12} {instance.get('_is_ambiguous', 'N/A'):<12}")
    
    print("="*80)
    
    # 创建新的dataset对象
    train_dataset = datasets.Dataset.from_list(train_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)
    minitest_dataset = datasets.Dataset.from_list(minitest_processed)
    tinytest_dataset = datasets.Dataset.from_list(tinytest_processed)
    
    # add a row to each data item that represents a unique id
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
                    'is_ambiguous': example.get('_is_ambiguous', False),
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
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    minitest_dataset.to_parquet(os.path.join(local_dir, 'minitest.parquet'))
    tinytest_dataset.to_parquet(os.path.join(local_dir, 'tinytest.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
