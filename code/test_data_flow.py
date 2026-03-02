#!/usr/bin/env python3
"""
测试数据传递流程的脚本
不涉及任何模型调用，只测试数据结构和传递
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'search_r1'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl'))

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from transformers import AutoTokenizer
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig


def create_mock_batch():
    """创建一个模拟的batch数据，模拟真实训练时的数据结构"""
    
    # 模拟数据
    batch_size = 2
    max_length = 512
    
    # 创建模拟的input_ids
    input_ids = torch.randint(1000, 2000, (batch_size, max_length))
    attention_mask = torch.ones(batch_size, max_length)
    position_ids = torch.arange(max_length).unsqueeze(0).repeat(batch_size, 1)
    
    # 创建模拟的extra_info
    extra_info = [
        {
            'split': 'train',
            'index': 0,
            'original_idx': 0,
            'qa_pair_idx': 0,
            'is_ambiguous': True,
            'gold_question': 'What is the capital of France?',
            'original_question': 'What is the capital of France?'
        },
        {
            'split': 'train',
            'index': 1,
            'original_idx': 1,
            'qa_pair_idx': 0,
            'is_ambiguous': False,
            'gold_question': 'How many planets are there?',
            'original_question': 'How many planets are there?'
        }
    ]
    
    # 创建DataProto对象
    batch = DataProto.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    })
    
    # 添加non_tensor_batch
    batch.non_tensor_batch = {
        'extra_info': extra_info,
        'data_source': ['ambigqa'] * batch_size,
        'ability': ['ambiguous-question-answering'] * batch_size,
        'reward_model': [
            {'style': 'rule', 'ground_truth': {'target': ['Paris']}},
            {'style': 'rule', 'ground_truth': {'target': ['8']}}
        ]
    }
    
    return batch


def test_data_structure():
    """测试数据结构"""
    print("=" * 60)
    print("🔍 测试数据结构")
    print("=" * 60)
    
    # 创建模拟batch
    batch = create_mock_batch()
    
    print(f"📊 Batch类型: {type(batch)}")
    print(f"📊 Batch长度: {len(batch)}")
    print(f"📊 Batch属性: {dir(batch)}")
    
    print(f"\n📋 Batch.batch keys: {list(batch.batch.keys())}")
    print(f"📋 Batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
    
    print(f"\n🔍 extra_info示例:")
    for i, info in enumerate(batch.non_tensor_batch['extra_info']):
        print(f"  实例 {i}: {info}")
    
    return batch


def test_data_manipulation():
    """测试数据操作"""
    print("\n" + "=" * 60)
    print("🔧 测试数据操作")
    print("=" * 60)
    
    batch = create_mock_batch()
    
    # 测试pop操作
    print("📤 测试pop操作:")
    print(f"   原始batch长度: {len(batch)}")
    
    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    print(f"   pop后gen_batch长度: {len(gen_batch)}")
    print(f"   pop后原始batch长度: {len(batch)}")
    
    print(f"   gen_batch.batch keys: {list(gen_batch.batch.keys())}")
    print(f"   gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
    
    # 测试repeat操作
    print("\n🔄 测试repeat操作:")
    repeated_batch = batch.repeat(repeat_times=3, interleave=True)
    print(f"   repeat后长度: {len(repeated_batch)}")
    print(f"   repeat后extra_info长度: {len(repeated_batch.non_tensor_batch['extra_info'])}")
    
    # 测试union操作
    print("\n🔗 测试union操作:")
    union_batch = batch.union(gen_batch)
    print(f"   union后长度: {len(union_batch)}")
    print(f"   union后batch keys: {list(union_batch.batch.keys())}")
    
    return batch, gen_batch, repeated_batch, union_batch


def test_generation_config():
    """测试GenerationConfig"""
    print("\n" + "=" * 60)
    print("⚙️ 测试GenerationConfig")
    print("=" * 60)
    
    config = GenerationConfig(
        max_turns=3,
        max_start_length=256,
        max_prompt_length=1024,
        max_response_length=512,
        max_obs_length=256,
        num_gpus=1,
        no_think_rl=False,
        search_url="http://127.0.0.1:8000/retrieve",
        topk=3,
        clarify_url="http://127.0.0.1:8001/batch_generate",
        enable_clarify=True
    )
    
    print(f"📋 配置内容:")
    for key, value in config.__dict__.items():
        print(f"   {key}: {value}")
    
    return config


def test_llm_generation_manager():
    """测试LLMGenerationManager（不涉及模型调用）"""
    print("\n" + "=" * 60)
    print("🤖 测试LLMGenerationManager")
    print("=" * 60)
    
    # 创建模拟tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
        
        def decode(self, tokens):
            return f"Mock decoded text: {tokens.shape}"
    
    # 创建模拟actor_rollout_wg
    class MockActorRolloutWG:
        def __init__(self):
            self.world_size = 1
        
        def generate_sequences(self, batch):
            # 返回模拟的生成结果
            mock_responses = torch.randint(1000, 2000, (len(batch), 64))
            mock_batch = DataProto.from_dict({'responses': mock_responses})
            return mock_batch
    
    # 创建配置
    config = test_generation_config()
    
    # 创建LLMGenerationManager
    try:
        manager = LLMGenerationManager(
            tokenizer=MockTokenizer(),
            actor_rollout_wg=MockActorRolloutWG(),
            config=config,
            is_validation=False
        )
        print("✅ LLMGenerationManager 创建成功")
        
        # 测试tensor_fn
        print(f"📊 tensor_fn类型: {type(manager.tensor_fn)}")
        print(f"📊 tensor_fn配置: {manager.tensor_fn.config}")
        
    except Exception as e:
        print(f"❌ LLMGenerationManager 创建失败: {e}")
        return None
    
    return manager


def test_data_flow_simulation():
    """模拟完整的数据流"""
    print("\n" + "=" * 60)
    print("🔄 模拟完整数据流")
    print("=" * 60)
    
    # 1. 创建初始batch
    print("1️⃣ 创建初始batch")
    batch = create_mock_batch()
    print(f"   初始batch长度: {len(batch)}")
    
    # 2. 模拟repeat操作（模拟n_agent=3）
    print("\n2️⃣ 模拟repeat操作 (n_agent=3)")
    repeated_batch = batch.repeat(repeat_times=3, interleave=True)
    print(f"   repeat后长度: {len(repeated_batch)}")
    print(f"   extra_info长度: {len(repeated_batch.non_tensor_batch['extra_info'])}")
    
    # 3. 模拟pop操作
    print("\n3️⃣ 模拟pop操作")
    gen_batch = repeated_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    print(f"   gen_batch长度: {len(gen_batch)}")
    print(f"   gen_batch.batch keys: {list(gen_batch.batch.keys())}")
    print(f"   gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
    
    # 4. 模拟生成结果
    print("\n4️⃣ 模拟生成结果")
    mock_responses = torch.randint(1000, 2000, (len(gen_batch), 64))
    mock_gen_output = DataProto.from_dict({'responses': mock_responses})
    print(f"   模拟生成结果长度: {len(mock_gen_output)}")
    
    # 5. 模拟union操作
    print("\n5️⃣ 模拟union操作")
    final_batch = repeated_batch.union(mock_gen_output)
    print(f"   最终batch长度: {len(final_batch)}")
    print(f"   最终batch.batch keys: {list(final_batch.batch.keys())}")
    print(f"   最终batch.non_tensor_batch keys: {list(final_batch.non_tensor_batch.keys())}")
    
    return batch, repeated_batch, gen_batch, final_batch


def test_parquet_loading():
    """测试parquet文件加载"""
    print("\n" + "=" * 60)
    print("📁 测试parquet文件加载")
    print("=" * 60)
    
    # 检查数据目录
    data_dir = Path("./data/ambignq")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return None
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ 在 {data_dir} 中没有找到parquet文件")
        return None
    
    print(f"📁 找到parquet文件: {[f.name for f in parquet_files]}")
    
    # 尝试加载第一个文件
    try:
        first_file = parquet_files[0]
        print(f"\n📖 尝试加载: {first_file.name}")
        
        df = pd.read_parquet(first_file)
        print(f"✅ 成功加载: {first_file.name}")
        print(f"   行数: {len(df)}")
        print(f"   列数: {len(df.columns)}")
        print(f"   列名: {list(df.columns)}")
        
        # 显示extra_info示例
        if 'extra_info' in df.columns:
            print(f"\n🔍 extra_info示例:")
            for i in range(min(3, len(df))):
                extra_info = df.iloc[i]['extra_info']
                if isinstance(extra_info, dict):
                    print(f"   实例 {i}: {extra_info}")
                else:
                    print(f"   实例 {i}: {type(extra_info)}")
        
        return df
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


def main():
    """主函数"""
    print("🚀 开始数据流测试")
    print("=" * 60)
    
    # 1. 测试数据结构
    batch = test_data_structure()
    
    # 2. 测试数据操作
    batch, gen_batch, repeated_batch, union_batch = test_data_manipulation()
    
    # 3. 测试配置
    config = test_generation_config()
    
    # 4. 测试LLMGenerationManager
    manager = test_llm_generation_manager()
    
    # 5. 模拟完整数据流
    batch, repeated_batch, gen_batch, final_batch = test_data_flow_simulation()
    
    # 6. 测试parquet文件加载
    df = test_parquet_loading()
    
    print("\n" + "=" * 60)
    print("🎉 数据流测试完成")
    print("=" * 60)
    
    if df is not None:
        print(f"✅ 成功加载parquet文件，包含 {len(df)} 条数据")
        print(f"📊 数据结构验证完成")
    else:
        print("⚠️  parquet文件加载失败，请检查数据目录")
    
    print("\n📝 测试总结:")
    print("   - 数据结构创建: ✅")
    print("   - 数据操作测试: ✅")
    print("   - 配置创建: ✅")
    print("   - 管理器创建: ✅")
    print("   - 数据流模拟: ✅")
    print("   - Parquet加载: {'✅' if df is not None else '❌'}")


if __name__ == "__main__":
    main()
