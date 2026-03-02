#!/usr/bin/env python3
"""
快速CoT测试脚本
用于快速验证CoT提示和OpenAI ChatGPT API的集成
"""

import os
import sys
import json
import time
import openai
from typing import Dict, Any

# 添加baseline目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
from cot_prompt import create_test_prompt, parse_cot_response

def test_cot_integration():
    """测试CoT集成"""
    print("🚀 开始CoT集成测试...")
    
    # 检查API配置（优先使用Azure OpenAI）
    azure_endpoint = os.getenv('AZURE_ENDPOINT', "https://cpii-s5.openai.azure.com/")
    azure_api_key = os.getenv('AZURE_API_KEY', '91e5ea9bf61c4769a44b0b0b5c67d559')
    azure_deployment = os.getenv('AZURE_DEPLOYMENT', 'gpt-4o')
    azure_api_version = os.getenv('AZURE_API_VERSION', '2024-02-01')
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # 设置OpenAI客户端
    if azure_endpoint and azure_api_key:
        print("🔵 使用Azure OpenAI服务")
        client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version
        )
        model_name = azure_deployment
    elif openai_api_key:
        print("🔵 使用OpenAI服务")
        client = openai.OpenAI(api_key=openai_api_key)
        model_name = "gpt-3.5-turbo"
    else:
        print("❌ 请设置API配置")
        print("Azure OpenAI: 设置 AZURE_ENDPOINT, AZURE_API_KEY, AZURE_DEPLOYMENT")
        print("OpenAI: 设置 OPENAI_API_KEY")
        return False
    
    # 测试API连接
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ API连接正常 (模型: {model_name})")
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False
    
    # 测试用例
    test_cases = [
        {
            "question": "What is Python?",
            "document": "Python is a programming language created by Guido van Rossum.",
            "conversation_history": "User: I want to learn programming.",
            "expected_ambiguous": True
        },
        {
            "question": "What is the capital of France?",
            "document": "France is a country in Europe. Its capital city is Paris.",
            "conversation_history": "",
            "expected_ambiguous": False
        },
        {
            "question": "How do I cook rice?",
            "document": "Rice cooking methods vary by type and equipment.",
            "conversation_history": "",
            "expected_ambiguous": True
        }
    ]
    
    print(f"\n📝 运行 {len(test_cases)} 个测试用例...")
    
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1} ---")
        print(f"问题: {test_case['question']}")
        
        # 创建CoT提示
        cot_prompt = create_test_prompt(
            test_case['question'],
            test_case['document'],
            test_case['conversation_history']
        )
        
        # 调用API
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing question ambiguity and providing clear responses."},
                    {"role": "user", "content": cot_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content
            
            # 解析CoT响应
            parsed = parse_cot_response(llm_response)
            
            # 检查结果
            is_correct = parsed['is_ambiguous'] == test_case['expected_ambiguous']
            status = "✅" if is_correct else "❌"
            
            print(f"{status} CoT判断: {'模糊' if parsed['is_ambiguous'] else '不模糊'}")
            print(f"期望: {'模糊' if test_case['expected_ambiguous'] else '不模糊'}")
            
            if parsed['is_ambiguous']:
                print(f"澄清问题: {parsed['clarifying_question'][:100]}...")
            else:
                print(f"答案: {parsed['answer'][:100]}...")
            
            results.append({
                'test_case': i+1,
                'question': test_case['question'],
                'expected_ambiguous': test_case['expected_ambiguous'],
                'detected_ambiguous': parsed['is_ambiguous'],
                'correct': is_correct,
                'raw_response': llm_response,
                'parsed_response': parsed
            })
            
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            results.append({
                'test_case': i+1,
                'question': test_case['question'],
                'error': str(e)
            })
        
        # 避免API调用过于频繁
        if i < len(test_cases) - 1:
            time.sleep(1)
    
    # 生成测试摘要
    print(f"\n📊 测试摘要")
    print("=" * 50)
    
    successful_tests = [r for r in results if 'error' not in r]
    correct_predictions = [r for r in successful_tests if r.get('correct', False)]
    
    print(f"总测试数: {len(test_cases)}")
    print(f"成功测试: {len(successful_tests)}")
    print(f"正确预测: {len(correct_predictions)}")
    print(f"准确率: {len(correct_predictions)/len(successful_tests)*100:.1f}%" if successful_tests else "N/A")
    
    # 保存详细结果
    output_file = "cot_integration_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {output_file}")
    
    return len(correct_predictions) == len(successful_tests)

def test_dataset_samples():
    """测试数据集样本"""
    print("\n🔍 测试数据集样本...")
    
    # 检查API配置（优先使用Azure OpenAI）
    azure_endpoint = os.getenv('AZURE_ENDPOINT')
    azure_api_key = os.getenv('AZURE_API_KEY')
    azure_deployment = os.getenv('AZURE_DEPLOYMENT', 'gpt-4o')
    azure_api_version = os.getenv('AZURE_API_VERSION', '2024-02-01')
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # 设置OpenAI客户端
    if azure_endpoint and azure_api_key:
        print("🔵 使用Azure OpenAI服务")
        client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version
        )
        model_name = azure_deployment
    elif openai_api_key:
        print("🔵 使用OpenAI服务")
        client = openai.OpenAI(api_key=openai_api_key)
        model_name = "gpt-3.5-turbo"
    else:
        print("❌ 请设置API配置")
        return
    
    # 查找可用的数据集文件
    data_dir = "/mnt/users_home/cpii.local/yli/Ambig-R1/data"
    dataset_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json') and ('dev' in file or 'test' in file or 'validation' in file):
                dataset_files.append(os.path.join(root, file))
    
    if not dataset_files:
        print("❌ 未找到数据集文件")
        return
    
    print(f"找到 {len(dataset_files)} 个数据集文件")
    
    # 测试第一个数据集的前几个样本
    test_file = dataset_files[0]
    print(f"测试文件: {test_file}")
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 取前3个样本进行测试
        test_samples = data[:3] if isinstance(data, list) else []
        
        if not test_samples:
            print("❌ 数据集为空或格式不支持")
            return
        
        print(f"测试 {len(test_samples)} 个样本...")
        
        for i, sample in enumerate(test_samples):
            print(f"\n--- 样本 {i+1} ---")
            
            # 提取问题
            question = sample.get('question', '')
            if not question:
                print("❌ 样本中没有问题")
                continue
            
            print(f"问题: {question[:100]}...")
            
            # 使用CoT分析
            cot_prompt = create_test_prompt(question)
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing question ambiguity and providing clear responses."},
                        {"role": "user", "content": cot_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                llm_response = response.choices[0].message.content
                parsed = parse_cot_response(llm_response)
                
                print(f"CoT判断: {'模糊' if parsed['is_ambiguous'] else '不模糊'}")
                if parsed['is_ambiguous']:
                    print(f"澄清问题: {parsed['clarifying_question'][:100]}...")
                else:
                    print(f"答案: {parsed['answer'][:100]}...")
            
            except Exception as e:
                print(f"❌ API调用失败: {e}")
            
            time.sleep(1)
    
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")

def main():
    """主函数"""
    print("🧪 CoT快速测试脚本")
    print("=" * 50)
    
    # 运行集成测试
    success = test_cot_integration()
    
    if success:
        print("\n🎉 CoT集成测试通过！")
        
        # 运行数据集样本测试
        test_dataset_samples()
    else:
        print("\n⚠️ CoT集成测试失败，请检查配置")
    
    print("\n🏁 测试完成！")

if __name__ == "__main__":
    main()
