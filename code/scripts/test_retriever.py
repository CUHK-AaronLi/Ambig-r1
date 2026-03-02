#!/usr/bin/env python3
"""
测试 retrieval server 的简单脚本
"""

import requests
import json

def test_retriever(base_url="http://127.0.0.1:8000", debug=False, output_file="retrieval_test_results.txt"):
    """测试retriever服务"""
    
    # 测试查询列表 - 简化测试
    test_queries = [
        "machine learning",  # 简单查询
        "python",           # 简单查询
        "computer science", # 简单查询
    ]
    
    # 创建输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("🔍 测试 Retrieval Server\n")
        f.write("=" * 60 + "\n")
        f.write(f"Server URL: {base_url}\n")
        f.write(f"测试查询数量: {len(test_queries)}\n")
        f.write(f"测试时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        print("🔍 测试 Retrieval Server")
        print("=" * 60)
        print(f"Server URL: {base_url}")
        print(f"测试查询数量: {len(test_queries)}")
        print(f"结果将保存到: {output_file}")
        print()
    
    # 测试每个查询
    for i, query in enumerate(test_queries, 1):
        print(f"--- 查询 {i}: {query} ---")
        
        # 准备请求数据 - 只测试topk=3，简化输出
        request_data = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        
        # 打开文件进行追加写入
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"--- 查询 {i}: {query} ---\n")
            
            try:
                # 发送请求
                response = requests.post(
                    f"{base_url}/retrieve",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 适配不同的响应格式
                    if 'result' in result:
                        # 新格式：result 字段，结构是 [[{document, score}, ...]]
                        if isinstance(result['result'], list) and len(result['result']) > 0:
                            doc_results = result['result'][0]  # 取第一个查询的结果
                            result_count = len(doc_results) if isinstance(doc_results, list) else 0
                        else:
                            doc_results = []
                            result_count = 0
                    elif 'results' in result:
                        # 旧格式：results 字段
                        doc_results = result['results'][0] if result['results'] else []
                        result_count = len(doc_results)
                    else:
                        doc_results = []
                        result_count = 0
                    
                    success_msg = f"✅ 请求成功\n返回结果数量: {result_count}\n"
                    print(success_msg.strip())
                    f.write(success_msg)
                    
                    # 显示结果 - 一条一条输出
                    if doc_results:
                        f.write(f"\n📄 检索结果:\n")
                        print(f"\n📄 检索结果:")
                        
                        for k, item in enumerate(doc_results):
                            result_msg = f"\n  结果 {k+1}:\n"
                            f.write(result_msg)
                            print(result_msg.strip())
                            
                            # 处理新的响应格式：{document: {...}, score: ...}
                            if isinstance(item, dict) and 'document' in item:
                                doc = item['document']
                                score = item.get('score', 'N/A')
                                
                                # 显示分数
                                score_msg = f"    分数: {score:.4f}\n" if isinstance(score, (int, float)) else f"    分数: {score}\n"
                                f.write(score_msg)
                                print(score_msg.strip())
                                
                                # 显示文档信息
                                if 'title' in doc:
                                    title_msg = f"    标题: {doc['title']}\n"
                                    f.write(title_msg)
                                    print(title_msg.strip())
                                if 'text' in doc:
                                    text_preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
                                    text_msg = f"    内容: {text_preview}\n"
                                    f.write(text_msg)
                                    print(text_msg.strip())
                                if 'id' in doc:
                                    id_msg = f"    ID: {doc['id']}\n"
                                    f.write(id_msg)
                                    print(id_msg.strip())
                            else:
                                # 处理旧格式或其他格式
                                if 'title' in item:
                                    title_msg = f"    标题: {item['title']}\n"
                                    f.write(title_msg)
                                    print(title_msg.strip())
                                if 'text' in item:
                                    text_preview = item['text'][:150] + "..." if len(item['text']) > 150 else item['text']
                                    text_msg = f"    内容: {text_preview}\n"
                                    f.write(text_msg)
                                    print(text_msg.strip())
                                if 'score' in item:
                                    score_msg = f"    分数: {item['score']:.4f}\n"
                                    f.write(score_msg)
                                    print(score_msg.strip())
                    
                    # 显示相关性分数
                    if 'scores' in result and result['scores']:
                        scores = result['scores'][0]  # 第一个查询的分数
                        scores_msg = f"\n🎯 相关性分数: {[f'{s:.4f}' for s in scores]}\n"
                        f.write(scores_msg)
                        print(scores_msg.strip())
                    else:
                        no_scores_msg = f"\n⚠️  没有返回分数信息\n"
                        f.write(no_scores_msg)
                        print(no_scores_msg.strip())
                        
                    # 如果开启调试模式，写入完整响应
                    if debug:
                        debug_msg = f"\n🔍 完整响应:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n"
                        f.write(debug_msg)
                        
                else:
                    error_msg = f"❌ 请求失败: {response.status_code}\n错误信息: {response.text}\n"
                    print(error_msg.strip())
                    f.write(error_msg)
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"❌ 网络错误: {e}\n"
                print(error_msg.strip())
                f.write(error_msg)
            except Exception as e:
                error_msg = f"❌ 其他错误: {e}\n"
                print(error_msg.strip())
                f.write(error_msg)
                
            f.write("-" * 60 + "\n\n")
            
        print("-" * 60)
        print()

def test_batch_queries(base_url="http://127.0.0.1:8000"):
    """测试批量查询"""
    
    print("🔍 测试批量查询")
    print("=" * 60)
    
    # 批量查询
    batch_queries = [
        "machine learning algorithms",
        "deep neural networks",
        "natural language processing"
    ]
    
    request_data = {
        "queries": batch_queries,
        "topk": 2,
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
            print(f"✅ 批量请求成功")
            print(f"处理查询数量: {len(batch_queries)}")
            print(f"返回结果组数: {len(result.get('results', []))}")
            
            # 显示每个查询的结果
            for i, (query, doc_results) in enumerate(zip(batch_queries, result.get('results', []))):
                print(f"\n📝 查询 {i+1}: '{query}'")
                for j, doc in enumerate(doc_results):
                    print(f"  结果 {j+1}:")
                    if 'title' in doc:
                        print(f"    标题: {doc['title']}")
                    if 'text' in doc:
                        text_preview = doc['text'][:80] + "..." if len(doc['text']) > 80 else doc['text']
                        print(f"    内容: {text_preview}")
                    if 'score' in doc:
                        print(f"    分数: {doc['score']:.4f}")
                        
        else:
            print(f"❌ 批量请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 批量查询错误: {e}")

def check_server_status(base_url="http://127.0.0.1:8000"):
    """检查服务器状态"""
    
    print("🏥 检查服务器状态")
    print("=" * 60)
    
    try:
        # 尝试访问根路径或健康检查端点
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ 服务器响应: {response.status_code}")
        
        # 尝试访问可能的健康检查端点
        health_endpoints = ["/health", "/status", "/info"]
        for endpoint in health_endpoints:
            try:
                health_response = requests.get(f"{base_url}{endpoint}", timeout=5)
                print(f"✅ 健康检查 {endpoint}: {health_response.status_code}")
                if health_response.status_code == 200:
                    print(f"   响应内容: {health_response.text[:200]}...")
            except:
                print(f"❌ 健康检查 {endpoint}: 不可用")
                
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("请确保 retrieval server 正在运行在 http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试 Retrieval Server')
    parser.add_argument('--url', default='http://127.0.0.1:8000', help='服务器URL')
    parser.add_argument('--output', default='retrieval_test_results.txt', help='输出文件名')
    parser.add_argument('--debug', action='store_true', help='开启调试模式，显示完整响应')
    parser.add_argument('--skip-status-check', action='store_true', help='跳过服务器状态检查')
    
    args = parser.parse_args()
    
    print("🚀 Retrieval Server 测试工具")
    print("=" * 60)
    
    # 检查服务器状态
    if not args.skip_status_check:
        if not check_server_status(args.url):
            print("\n请先启动 retrieval server!")
            exit(1)
        print()
    
    # 测试单个查询
    test_retriever(base_url=args.url, debug=args.debug, output_file=args.output)
    
    print()
    
    # 测试批量查询
    test_batch_queries(args.url)
    
    print(f"🎉 测试完成! 结果已保存到: {args.output}")
