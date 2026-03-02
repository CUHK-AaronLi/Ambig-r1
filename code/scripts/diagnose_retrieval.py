#!/usr/bin/env python3
"""
诊断 retrieval server 问题的脚本
"""

import requests
import json
import time

def test_server_connection(base_url="http://127.0.0.1:8000"):
    """测试服务器连接"""
    print("🔍 诊断 Retrieval Server 连接问题")
    print("=" * 60)
    
    # 1. 测试基本连接
    print("1. 测试基本连接...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   ✅ 服务器响应: {response.status_code}")
        print(f"   响应内容: {response.text[:200]}...")
    except requests.exceptions.ConnectionError:
        print("   ❌ 无法连接到服务器")
        print("   💡 可能的原因:")
        print("      - Retrieval server 没有启动")
        print("      - 服务器运行在不同的端口")
        print("      - 防火墙阻止了连接")
        return False
    except Exception as e:
        print(f"   ❌ 连接错误: {e}")
        return False
    
    # 2. 测试健康检查端点
    print("\n2. 测试健康检查端点...")
    health_endpoints = ["/health", "/status", "/info", "/docs"]
    for endpoint in health_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"   ✅ {endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"      响应: {response.text[:100]}...")
        except:
            print(f"   ❌ {endpoint}: 不可用")
    
    # 3. 测试检索端点
    print("\n3. 测试检索端点...")
    test_queries = [
        "test",
        "hello",
        "machine learning"
    ]
    
    for query in test_queries:
        print(f"\n   测试查询: '{query}'")
        request_data = {
            "queries": [query],
            "topk": 1,
            "return_scores": True
        }
        
        try:
            response = requests.post(
                f"{base_url}/retrieve",
                json=request_data,
                timeout=10
            )
            
            print(f"   状态码: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 请求成功")
                print(f"   结果数量: {len(result.get('results', []))}")
                
                # 显示完整响应
                print(f"   完整响应:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                if len(result.get('results', [])) == 0:
                    print("   ⚠️  警告: 返回结果为空!")
                    print("   💡 可能的原因:")
                    print("      - 数据库/索引没有加载")
                    print("      - 查询与数据库内容不匹配")
                    print("      - 索引文件损坏或路径错误")
                
            else:
                print(f"   ❌ 请求失败")
                print(f"   错误信息: {response.text}")
                
        except Exception as e:
            print(f"   ❌ 请求异常: {e}")
    
    return True

def check_system_resources():
    """检查系统资源"""
    print("\n🔧 检查系统资源")
    print("=" * 60)
    
    try:
        import psutil
        
        # 检查内存使用
        memory = psutil.virtual_memory()
        print(f"内存使用: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
        
        # 检查CPU使用
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU使用: {cpu_percent}%")
        
        # 检查GPU使用
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU可用: {torch.cuda.device_count()} 个设备")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("GPU不可用")
        except ImportError:
            print("PyTorch未安装，无法检查GPU")
            
    except ImportError:
        print("psutil未安装，无法检查系统资源")

if __name__ == "__main__":
    print("🚀 Retrieval Server 诊断工具")
    print("=" * 60)
    
    # 检查系统资源
    check_system_resources()
    
    # 测试服务器连接
    test_server_connection()
    
    print("\n🎯 诊断建议:")
    print("1. 如果无法连接服务器，请检查:")
    print("   - 服务器是否已启动: ps aux | grep retrieval")
    print("   - 端口是否正确: netstat -tlnp | grep 8000")
    print("   - 防火墙设置")
    print()
    print("2. 如果连接成功但返回空结果，请检查:")
    print("   - 数据库/索引文件是否存在")
    print("   - 索引文件是否正确加载")
    print("   - 查询内容是否与数据库匹配")
    print()
    print("3. 如果服务器响应慢，请检查:")
    print("   - 系统资源使用情况")
    print("   - GPU内存是否充足")
    print("   - 网络延迟")

