#!/usr/bin/env python3
"""
测试修改后的代码
"""

import requests
import json

def test_modified():
    """测试修改后的代码"""
    base_url = "http://127.0.0.1:8000"
    
    print("🔍 测试修改后的代码")
    print("=" * 50)
    
    # 测试查询
    query = "machine learning"
    print(f"测试查询: {query}")
    
    request_data = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }
    
    try:
        response = requests.post(
            f"{base_url}/retrieve",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 请求成功!")
            print("完整响应:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 使用修改后的逻辑
            print("\n📊 使用修改后的逻辑分析:")
            
            # 适配不同的响应格式
            if 'result' in result:
                # 新格式：result 字段
                doc_results = result['result']
                result_count = len(doc_results) if isinstance(doc_results, list) else 0
                print(f"- 使用 'result' 字段")
                print(f"- result 类型: {type(doc_results)}")
                print(f"- result 长度: {result_count}")
                
                if doc_results and isinstance(doc_results, list):
                    print(f"- 第一个结果键: {list(doc_results[0].keys()) if isinstance(doc_results[0], dict) else 'N/A'}")
                    
            elif 'results' in result:
                # 旧格式：results 字段
                doc_results = result['results'][0] if result['results'] else []
                result_count = len(doc_results)
                print(f"- 使用 'results' 字段")
                print(f"- results 长度: {result_count}")
            else:
                doc_results = []
                result_count = 0
                print("- 没有找到 result 或 results 字段!")
            
            print(f"\n最终结果数量: {result_count}")
            
            if doc_results:
                print("\n📄 检索结果:")
                for k, doc in enumerate(doc_results):
                    print(f"\n  结果 {k+1}:")
                    if isinstance(doc, dict):
                        for key, value in doc.items():
                            if key == 'text' and len(str(value)) > 100:
                                print(f"    {key}: {str(value)[:100]}...")
                            else:
                                print(f"    {key}: {value}")
                    else:
                        print(f"    内容: {doc}")
            else:
                print("\n⚠️  没有检索结果!")
                
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 异常: {e}")

if __name__ == "__main__":
    test_modified()

