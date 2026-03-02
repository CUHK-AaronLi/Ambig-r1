#!/usr/bin/env python3
"""
检查 train.parquet 文件的数据结构和内容
"""

import pandas as pd
import os
import sys
import json

def check_parquet_data(file_path):
    """检查parquet文件的数据结构和内容"""
    
    print(f"正在检查文件: {file_path}")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    try:
        # 读取parquet文件
        print("📖 正在读取parquet文件...")
        df = pd.read_parquet(file_path)
        
        print(f"✅ 成功读取数据")
        print(f"📊 数据形状: {df.shape}")
        print(f"🔢 行数: {len(df)}")
        print(f"📋 列数: {len(df.columns)}")
        
        print("\n📋 列名列表:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print("\n🔍 随机5条数据预览:")
        print("=" * 40)
        # 随机采样5条数据
        sample_df = df.sample(n=min(5, len(df)), random_state=42)
        
        for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
            print(f"\n--- 数据 {idx} ---")
            for col in df.columns:
                value = row[col]
                if isinstance(value, str) and len(value) > 100:
                    print(f"{col}: {value[:100]}...")
                else:
                    print(f"{col}: {value}")

        print("\n🔍 随机数据的extra_info列详细内容:")
        print("=" * 40)
        if 'extra_info' in df.columns:
            for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
                print(f"\n--- 数据 {idx} 的 extra_info ---")
                extra_info = row['extra_info']
                if isinstance(extra_info, dict):
                    for key, value in extra_info.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"{key}: {value[:100]}...")
                        else:
                            print(f"{key}: {value}")
                else:
                    print(f"extra_info类型: {type(extra_info)}")
                    print(f"内容: {extra_info}")
        else:
            print("❌ 没有找到extra_info列")
        
        # 保存随机5条数据到文件
        sample_file = "random_5_samples.txt"
        print(f"\n💾 保存随机5条数据到 {sample_file}")
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("=== 随机5条数据样本 ===\n\n")
            for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
                f.write(f"--- 数据 {idx} ---\n")
                for col in df.columns:
                    value = row[col]
                    f.write(f"{col}: {value}\n")
                f.write("\n")
        
        print(f"✅ 样本数据已保存到 {sample_file}")
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    
    # 默认数据目录
    default_data_dir = "data/ambignq"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 尝试多个可能的路径
        possible_paths = [
            f"{default_data_dir}/train.parquet",
            "data/train.parquet",
            "train.parquet",
            "../data/train.parquet",
            "../../data/train.parquet"
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            print("❌ 未找到 train.parquet 文件")
            print("请提供文件路径作为参数，或者确保文件在以下位置之一:")
            for path in possible_paths:
                print(f"  - {path}")
            return
    
    # 检查数据
    check_parquet_data(file_path)

if __name__ == "__main__":
    main()
