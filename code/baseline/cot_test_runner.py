#!/usr/bin/env python3
"""
基于CoT提示的模糊问题检测测试脚本
集成CoT提示模板和OpenAI ChatGPT API，用于测试多个数据集
"""

import os
import json
import argparse
import time
import sys
import openai
from typing import List, Dict, Any
from pathlib import Path

# 添加baseline目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
from cot_prompt import create_test_prompt, parse_cot_response, format_cot_prompt

class CoTTestRunner:
    """CoT测试运行器"""
    
    def __init__(self, model: str = None, api_key: str = None, azure_endpoint: str = None, 
                 azure_deployment: str = None, azure_api_version: str = None):
        # 优先使用Azure OpenAI配置
        self.azure_endpoint = azure_endpoint or os.getenv('AZURE_ENDPOINT', "https://cpii-s5.openai.azure.com/")
        self.azure_api_key = api_key or os.getenv('AZURE_API_KEY', '91e5ea9bf61c4769a44b0b0b5c67d559')
        self.azure_deployment = azure_deployment or os.getenv('AZURE_DEPLOYMENT', 'gpt-4o')
        self.azure_api_version = azure_api_version or os.getenv('AZURE_API_VERSION', '2024-02-01')
        
        # OpenAI配置
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # 设置客户端和模型
        if self.azure_endpoint and self.azure_api_key:
            print("🔵 使用Azure OpenAI服务")
            self.client = openai.AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version
            )
            self.model = model or self.azure_deployment
            self.api_type = "azure"
        elif self.openai_api_key:
            print("🔵 使用OpenAI服务")
            self.client = openai.OpenAI(api_key=self.openai_api_key)
            self.model = model or "gpt-3.5-turbo"
            self.api_type = "openai"
        else:
            raise ValueError("请设置API配置：Azure OpenAI (AZURE_ENDPOINT, AZURE_API_KEY) 或 OpenAI (OPENAI_API_KEY)")
        
        self.results = []
        
    def test_api_health(self) -> bool:
        """测试API健康状态"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print(f"✅ OpenAI API健康检查通过")
            return True
        except Exception as e:
            print(f"❌ OpenAI API健康检查失败: {e}")
            return False
    
    def call_llm_api(self, prompt: str) -> str:
        """调用OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing question ambiguity and providing clear responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return f"[错误: {str(e)}]"
    
    def analyze_question_with_cot(self, question: str, document: str = "", conversation_history: str = "") -> Dict[str, Any]:
        """使用CoT方法分析问题"""
        # 创建CoT提示
        cot_prompt = create_test_prompt(question, document, conversation_history)
        
        # 调用LLM API
        response = self.call_llm_api(cot_prompt)
        
        # 解析响应
        parsed_response = parse_cot_response(response)
        
        return {
            'question': question,
            'document': document,
            'conversation_history': conversation_history,
            'cot_prompt': cot_prompt,
            'raw_response': response,
            'parsed_response': parsed_response,
            'timestamp': time.time()
        }
    
    def load_dataset(self, dataset_path: str, dataset_type: str = "auto") -> List[Dict]:
        """加载数据集"""
        print(f"加载数据集: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 根据数据集类型处理数据
        if dataset_type == "auto":
            dataset_type = self._detect_dataset_type(data)
        
        processed_data = self._process_dataset(data, dataset_type)
        print(f"数据集类型: {dataset_type}")
        print(f"加载了 {len(processed_data)} 个样本")
        
        return processed_data
    
    def _detect_dataset_type(self, data: Any) -> str:
        """自动检测数据集类型"""
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            if 'question' in sample and 'answer' in sample:
                if 'table' in sample:
                    return "pacific"
                elif 'used_queries' in sample:
                    return "ambignq"
                else:
                    return "coqa"
            elif 'question' in sample and 'clarification_question' in sample:
                return "clarification"
        
        return "unknown"
    
    def _process_dataset(self, data: Any, dataset_type: str) -> List[Dict]:
        """根据数据集类型处理数据"""
        processed = []
        
        if dataset_type == "ambignq":
            for item in data:
                processed.append({
                    'id': item.get('id', ''),
                    'question': item.get('question', ''),
                    'document': self._extract_document_from_ambignq(item),
                    'conversation_history': '',
                    'ground_truth': item.get('annotations', [])
                })
        
        elif dataset_type == "coqa":
            for item in data:
                processed.append({
                    'id': item.get('id', ''),
                    'question': item.get('question', ''),
                    'document': item.get('passage', ''),
                    'conversation_history': self._extract_conversation_history(item),
                    'ground_truth': item.get('answer', '')
                })
        
        elif dataset_type == "pacific":
            for item in data:
                processed.append({
                    'id': item.get('id', ''),
                    'question': item.get('question', ''),
                    'document': self._extract_document_from_pacific(item),
                    'conversation_history': '',
                    'ground_truth': item.get('answer', '')
                })
        
        else:
            # 通用处理
            if isinstance(data, list):
                for i, item in enumerate(data):
                    processed.append({
                        'id': item.get('id', f'sample_{i}'),
                        'question': item.get('question', ''),
                        'document': item.get('document', item.get('passage', '')),
                        'conversation_history': item.get('conversation_history', ''),
                        'ground_truth': item.get('answer', item.get('ground_truth', ''))
                    })
        
        return processed
    
    def _extract_document_from_ambignq(self, item: Dict) -> str:
        """从AmbigQA数据中提取文档"""
        if 'used_queries' in item and item['used_queries']:
            query = item['used_queries'][0]
            if 'results' in query and query['results']:
                snippets = [result.get('snippet', '') for result in query['results']]
                return ' '.join(snippets)
        return ''
    
    def _extract_conversation_history(self, item: Dict) -> str:
        """提取对话历史"""
        if 'additional_answers' in item:
            answers = item['additional_answers']
            if answers:
                return f"Previous answers: {'; '.join(answers)}"
        return ''
    
    def _extract_document_from_pacific(self, item: Dict) -> str:
        """从PACIFIC数据中提取文档"""
        if 'table' in item and 'table' in item['table']:
            table = item['table']['table']
            # 将表格转换为文本
            text_rows = []
            for row in table:
                text_rows.append(' | '.join(str(cell) for cell in row if cell))
            return '\n'.join(text_rows)
        return ''
    
    def run_tests(self, data: List[Dict], max_samples: int = None) -> List[Dict]:
        """运行测试"""
        if max_samples:
            data = data[:max_samples]
        
        print(f"开始运行CoT测试，共 {len(data)} 个样本...")
        
        for i, sample in enumerate(data):
            print(f"\n处理 {i+1}/{len(data)}: {sample['id']}")
            print(f"问题: {sample['question'][:100]}...")
            
            # 使用CoT分析问题
            result = self.analyze_question_with_cot(
                question=sample['question'],
                document=sample['document'],
                conversation_history=sample['conversation_history']
            )
            
            # 添加真实标签
            result['ground_truth'] = sample['ground_truth']
            result['sample_id'] = sample['id']
            
            self.results.append(result)
            
            # 显示结果
            parsed = result['parsed_response']
            print(f"CoT判断: {'模糊' if parsed['is_ambiguous'] else '不模糊'}")
            if parsed['is_ambiguous']:
                print(f"澄清问题: {parsed['clarifying_question'][:100]}...")
            else:
                print(f"答案: {parsed['answer'][:100]}...")
            
            # 避免API调用过于频繁
            if i < len(data) - 1:
                time.sleep(1)
        
        return self.results
    
    def save_results(self, output_file: str):
        """保存测试结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n测试结果已保存到: {output_file}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成测试摘要"""
        total = len(self.results)
        ambiguous_count = sum(1 for r in self.results if r['parsed_response']['is_ambiguous'])
        unambiguous_count = total - ambiguous_count
        
        return {
            'total_samples': total,
            'ambiguous_detected': ambiguous_count,
            'unambiguous_detected': unambiguous_count,
            'ambiguous_rate': ambiguous_count / total if total > 0 else 0,
            'model': self.model,
            'timestamp': time.time()
        }

def main():
    parser = argparse.ArgumentParser(description="基于CoT提示的模糊问题检测测试")
    parser.add_argument('--dataset_path', type=str, required=True, help="数据集文件路径")
    parser.add_argument('--dataset_type', type=str, default="auto", 
                       choices=["auto", "ambignq", "coqa", "pacific"], help="数据集类型")
    parser.add_argument('--model', type=str, default=None, help="模型名称（可选，优先使用环境变量）")
    parser.add_argument('--api_key', type=str, default=None, help="API密钥（可选，优先使用环境变量）")
    parser.add_argument('--azure_endpoint', type=str, default=None, help="Azure OpenAI端点（可选，优先使用环境变量）")
    parser.add_argument('--azure_deployment', type=str, default=None, help="Azure OpenAI部署名称（可选，优先使用环境变量）")
    parser.add_argument('--azure_api_version', type=str, default=None, help="Azure API版本（可选，优先使用环境变量）")
    parser.add_argument('--output', type=str, default="cot_test_results.json", help="输出文件名")
    parser.add_argument('--max_samples', type=int, default=None, help="最大测试样本数")
    
    args = parser.parse_args()
    
    # 创建测试运行器
    try:
        runner = CoTTestRunner(
            model=args.model,
            api_key=args.api_key,
            azure_endpoint=args.azure_endpoint,
            azure_deployment=args.azure_deployment,
            azure_api_version=args.azure_api_version
        )
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # 测试API健康状态
    print("=== 测试API ===")
    if not runner.test_api_health():
        print("❌ API不可用，请检查API配置和网络连接")
        return
    
    # 加载数据集
    print(f"\n=== 加载数据集 ===")
    try:
        data = runner.load_dataset(args.dataset_path, args.dataset_type)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 运行测试
    print(f"\n=== 运行CoT测试 ===")
    results = runner.run_tests(data, args.max_samples)
    
    # 保存结果
    print(f"\n=== 保存结果 ===")
    runner.save_results(args.output)
    
    # 生成摘要
    summary = runner.generate_summary()
    print(f"\n=== 测试摘要 ===")
    print(f"总样本数: {summary['total_samples']}")
    print(f"检测为模糊: {summary['ambiguous_detected']}")
    print(f"检测为不模糊: {summary['unambiguous_detected']}")
    print(f"模糊率: {summary['ambiguous_rate']:.2%}")
    
    # 保存摘要
    summary_file = args.output.replace('.json', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"摘要已保存到: {summary_file}")

if __name__ == "__main__":
    main()
