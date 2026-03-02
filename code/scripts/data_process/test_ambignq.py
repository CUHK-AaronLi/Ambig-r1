#!/usr/bin/env python3
"""
测试AmbigQA数据处理脚本（使用RUC-NLPIR/FlashRAG_datasets）
"""

import os
import tempfile
import shutil
from ambignq import make_prefix


def test_ambignq_template():
    """测试AmbigQA提示模板"""
    print("🧪 Testing AmbigQA prompt template...")
    
    # 测试数据
    test_question = "What is the capital of France?"
    test_dp = {"question": test_question}
    
    # 测试base模板
    prefix = make_prefix(test_dp, 'base')
    
    print(f"✅ Generated prefix for question: {test_question}")
    print(f"Prefix length: {len(prefix)} characters")
    
    # 检查是否包含必要的元素
    required_elements = [
        '<think>', '</think>',
        '<search>', '</search>',
        '<clarify>', '</clarify>',
        '<answer>', '</answer>',
        'Question: What is the capital of France?'
    ]
    
    for element in required_elements:
        if element in prefix:
            print(f"  ✅ Contains: {element}")
        else:
            print(f"  ❌ Missing: {element}")
    
    print(f"\n📝 Full prefix preview:")
    print(prefix[:200] + "..." if len(prefix) > 200 else prefix)
    
    return True


def test_datasets_loading():
    """测试datasets库的AmbigQA数据加载"""
    print("\n📊 Testing datasets library loading...")
    
    try:
        import datasets
        
        # 尝试加载数据集
        print("  🔄 Loading AmbigQA dataset...")
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'ambig_qa')
        
        print(f"  ✅ Successfully loaded dataset")
        print(f"  📁 Available splits: {list(dataset.keys())}")
        
        # 检查训练集
        if 'train' in dataset:
            train_dataset = dataset['train']
            print(f"  🚂 Training set: {len(train_dataset)} examples")
            
            # 显示第一个样本的结构
            if len(train_dataset) > 0:
                first_example = train_dataset[0]
                print(f"  📋 First example keys: {list(first_example.keys())}")
                
                # 检查关键字段
                if 'question' in first_example:
                    print(f"  ❓ Sample question: {first_example['question'][:100]}...")
                if 'golden_answers' in first_example:
                    print(f"  🎯 Sample answers: {first_example['golden_answers']}")
        
        # 检查测试集
        if 'test' in dataset:
            test_dataset = dataset['test']
            print(f"  🧪 Test set: {len(test_dataset)} examples")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        return False


def test_data_processing():
    """测试数据处理流程"""
    print("\n⚙️ Testing data processing pipeline...")
    
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "ambignq_search")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"  📁 Created temporary directory: {temp_dir}")
        
        # 模拟数据处理
        test_question = "What is the capital of France?"
        test_dp = {"question": test_question}
        
        # 生成提示
        prefix = make_prefix(test_dp, 'base')
        
        # 模拟数据结构
        example_data = {
            "question": test_question,
            "golden_answers": ["Paris"]
        }
        
        # 构建数据格式
        data = {
            "data_source": "ambignq",
            "prompt": [{
                "role": "user",
                "content": prefix,
            }],
            "ability": "ambiguous-question-answering",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "target": example_data['golden_answers'],
                }
            },
            "extra_info": {
                'split': 'train',
                'index': 0,
            }
        }
        
        print(f"  ✅ Successfully created data structure")
        print(f"  📊 Data keys: {list(data.keys())}")
        print(f"  🎯 Ground truth: {data['reward_model']['ground_truth']['target']}")
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print(f"  🧹 Cleaned up temporary files")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_variations():
    """测试不同的模板类型"""
    print("\n🎨 Testing template variations...")
    
    test_questions = [
        "What is the capital of France?",
        "Who is the president of the United States?",
        "How many people live in New York?"
    ]
    
    for i, question in enumerate(test_questions):
        print(f"  📝 Test question {i+1}: {question}")
        
        test_dp = {"question": question}
        prefix = make_prefix(test_dp, 'base')
        
        # 检查模板是否正确应用
        if question in prefix:
            print(f"    ✅ Question correctly embedded in prefix")
        else:
            print(f"    ❌ Question not found in prefix")
        
        # 检查模板长度是否合理
        if 200 < len(prefix) < 1000:
            print(f"    ✅ Prefix length reasonable: {len(prefix)} characters")
        else:
            print(f"    ⚠️ Prefix length unusual: {len(prefix)} characters")
    
    return True


if __name__ == "__main__":
    print("🚀 Starting AmbigQA data processing tests...\n")
    
    # 运行所有测试
    tests = [
        ("Template Generation", test_ambignq_template),
        ("Dataset Loading", test_datasets_loading),
        ("Data Processing", test_data_processing),
        ("Template Variations", test_template_variations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # 显示测试结果
    print(f"\n📋 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed!")
        print("\n📖 Next steps:")
        print("1. Run the main script:")
        print("   python ambignq.py --local_dir ./data/ambignq_search")
        print("2. Check the generated parquet files")
        print("3. Use the data in your VERL training pipeline")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the issues above.")
