#!/usr/bin/env python3
"""
CoT测试脚本使用示例
演示如何使用CoT提示和OpenAI API进行模糊问题检测
"""

import os
import sys
import json

# 添加baseline目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))

def demo_cot_prompt():
    """演示CoT提示的使用"""
    print("=== CoT提示演示 ===")
    
    try:
        from cot_prompt import create_test_prompt, parse_cot_response
        
        # 示例问题
        test_question = "What is Python?"
        test_document = "Python is a programming language created by Guido van Rossum."
        test_history = "User: I want to learn programming."
        
        # 创建CoT提示
        prompt = create_test_prompt(test_question, test_document, test_history)
        print("CoT提示:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        
        # 模拟响应解析
        mock_response = """Looking at this question, I need to analyze its ambiguity. The question "What is Python?" could refer to either the programming language or the snake species. Without additional context, this question is ambiguous because "Python" has multiple meanings.

Therefore, the question is ambiguous. The clarifying question is: Do you mean the Python programming language or the Python snake species?"""
        
        parsed = parse_cot_response(mock_response)
        print("\n解析结果:")
        print(f"是否模糊: {parsed['is_ambiguous']}")
        print(f"澄清问题: {parsed['clarifying_question']}")
        print(f"分析: {parsed['analysis']}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保cot-prompt.py文件存在且语法正确")

def demo_api_setup():
    """演示API设置"""
    print("\n=== API设置演示 ===")
    
    # 检查Azure OpenAI配置
    azure_endpoint = os.getenv('AZURE_ENDPOINT')
    azure_api_key = os.getenv('AZURE_API_KEY')
    azure_deployment = os.getenv('AZURE_DEPLOYMENT')
    azure_api_version = os.getenv('AZURE_API_VERSION')
    
    # 检查OpenAI配置
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if azure_endpoint and azure_api_key:
        print("✅ 找到Azure OpenAI配置")
        print(f"   端点: {azure_endpoint}")
        print(f"   部署: {azure_deployment}")
        print(f"   API版本: {azure_api_version}")
        print(f"   密钥长度: {len(azure_api_key)} 字符")
        print("   密钥前缀: " + azure_api_key[:8] + "...")
    elif openai_api_key:
        print("✅ 找到OpenAI配置")
        print(f"   密钥长度: {len(openai_api_key)} 字符")
        print("   密钥前缀: " + openai_api_key[:8] + "...")
    else:
        print("❌ 未找到API配置")
        print("请设置以下环境变量之一:")
        print("Azure OpenAI:")
        print("  export AZURE_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_API_KEY='your-api-key-here'")
        print("  export AZURE_DEPLOYMENT='gpt-4o'")
        print("  export AZURE_API_VERSION='2024-02-01'")
        print("OpenAI:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\n或者运行: source setup_azure.sh")

def demo_dataset_structure():
    """演示数据集结构"""
    print("\n=== 数据集结构演示 ===")
    
    # 查找数据集文件
    data_dir = "/mnt/users_home/cpii.local/yli/Ambig-R1/data"
    dataset_files = []
    
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json') and ('dev' in file or 'test' in file or 'validation' in file):
                    dataset_files.append(os.path.join(root, file))
    
    if dataset_files:
        print(f"找到 {len(dataset_files)} 个数据集文件:")
        for i, file_path in enumerate(dataset_files[:5]):  # 只显示前5个
            print(f"  {i+1}. {file_path}")
        
        if len(dataset_files) > 5:
            print(f"  ... 还有 {len(dataset_files) - 5} 个文件")
    else:
        print("❌ 未找到数据集文件")
        print(f"请检查数据目录: {data_dir}")

def demo_usage_examples():
    """演示使用示例"""
    print("\n=== 使用示例 ===")
    
    print("1. 快速测试:")
    print("   python quick_cot_test.py")
    
    print("\n2. 测试特定数据集:")
    print("   python cot_test_runner.py --dataset_path /path/to/dataset.json --max_samples 10")
    
    print("\n3. 使用Azure OpenAI:")
    print("   source setup_azure.sh")
    print("   python cot_test_runner.py --dataset_path /path/to/dataset.json")
    
    print("\n4. 使用OpenAI:")
    print("   export OPENAI_API_KEY='your-key'")
    print("   python cot_test_runner.py --dataset_path /path/to/dataset.json --model gpt-4")
    
    print("\n5. 指定数据集类型:")
    print("   python cot_test_runner.py --dataset_path /path/to/ambignq.json --dataset_type ambignq")

def main():
    """主函数"""
    print("🧪 CoT测试脚本使用演示")
    print("=" * 60)
    
    # 运行各个演示
    demo_cot_prompt()
    demo_api_setup()
    demo_dataset_structure()
    demo_usage_examples()
    
    print("\n" + "=" * 60)
    print("📚 更多信息请查看 README.md")
    print("🏁 演示完成！")

if __name__ == "__main__":
    main()
