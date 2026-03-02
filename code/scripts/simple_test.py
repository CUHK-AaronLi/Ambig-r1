#!/usr/bin/env python3
"""
直接测试 retrieval server 的简单脚本
"""

import requests
import json

def test_simple():
    """简单测试"""
    base_url = "http://127.0.0.1:8000"
    
    print("🔍 直接测试 Retrieval Server")
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
        print("发送请求...")
        response = requests.post(
            f"{base_url}/retrieve",
            json=request_data,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 请求成功!")
            print("完整响应:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 分析响应结构
            print("\n📊 响应分析:")
            print(f"- 响应键: {list(result.keys())}")
            
            # 检查不同的响应格式
            if 'result' in result:
                print(f"- result 类型: {type(result['result'])}")
                if isinstance(result['result'], list):
                    print(f"- result 长度: {len(result['result'])}")
                    if result['result']:
                        print(f"- 第一个结果类型: {type(result['result'][0])}")
                        print(f"- 第一个结果键: {list(result['result'][0].keys()) if isinstance(result['result'][0], dict) else 'N/A'}")
                else:
                    print(f"- result 内容: {result['result']}")
            elif 'results' in result:
                print(f"- results 类型: {type(result['results'])}")
                print(f"- results 长度: {len(result['results'])}")
                
                if result['results']:
                    print(f"- 第一个结果类型: {type(result['results'][0])}")
                    print(f"- 第一个结果长度: {len(result['results'][0])}")
                else:
                    print("- results 为空!")
            else:
                print("- 没有找到 result 或 results 字段!")
            
            if 'scores' in result:
                print(f"- scores 类型: {type(result['scores'])}")
                print(f"- scores 长度: {len(result['scores'])}")
            else:
                print("- 没有 scores 字段!")
                
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 异常: {e}")

if __name__ == "__main__":
    test_simple()
