#!/usr/bin/env python3
"""
检查数据文件中 clarification 相关字段是否存在
"""
import json
import sys
from pathlib import Path

def check_parquet_data(parquet_path: str, num_samples: int = 5):
    """检查 parquet 文件中的数据字段"""
    try:
        import pandas as pd
        
        print(f"Reading parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        print(f"\nTotal samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # 检查是否有 extra_info 列
        if 'extra_info' not in df.columns:
            print("\n❌ No 'extra_info' column found!")
            return
        
        print(f"\n{'='*80}")
        print("Checking first few samples:")
        print(f"{'='*80}")
        
        for idx in range(min(num_samples, len(df))):
            print(f"\n--- Sample {idx} ---")
            extra_info = df.iloc[idx]['extra_info']
            
            if isinstance(extra_info, dict):
                print(f"extra_info type: dict")
                print(f"extra_info keys: {list(extra_info.keys())}")
                
                # 检查关键字段
                key_fields = [
                    'clarification_turn_question',
                    'clarification_choice_text',
                    'clarification_reference',
                    '_is_ambiguous',
                    'clarification_choice_index',
                    'clarification_choice_total',
                ]
                
                for key in key_fields:
                    value = extra_info.get(key)
                    if value is None:
                        print(f"  {key}: None")
                    elif isinstance(value, dict):
                        print(f"  {key}: dict with keys: {list(value.keys())}")
                        if key == 'clarification_reference' and value:
                            print(f"    - clarification_question: '{value.get('clarification_question', 'EMPTY')[:100]}'")
                            print(f"    - clarification_user_answer: '{value.get('clarification_user_answer', 'EMPTY')[:100]}'")
                    elif isinstance(value, str):
                        print(f"  {key}: '{value[:100] if value else 'EMPTY'}'")
                    else:
                        print(f"  {key}: {type(value).__name__} = {value}")
            else:
                print(f"extra_info type: {type(extra_info)}")
                print(f"extra_info value: {str(extra_info)[:200]}")
        
        # 统计有多少样本有 clarification_reference
        print(f"\n{'='*80}")
        print("Statistics:")
        print(f"{'='*80}")
        
        has_clar_ref = 0
        has_clar_turn_q = 0
        has_clar_choice_t = 0
        is_ambiguous_count = 0
        
        for idx in range(len(df)):
            extra_info = df.iloc[idx]['extra_info']
            if isinstance(extra_info, dict):
                if extra_info.get('clarification_reference'):
                    has_clar_ref += 1
                if extra_info.get('clarification_turn_question'):
                    has_clar_turn_q += 1
                if extra_info.get('clarification_choice_text'):
                    has_clar_choice_t += 1
                if extra_info.get('_is_ambiguous'):
                    is_ambiguous_count += 1
        
        print(f"Samples with clarification_reference: {has_clar_ref}/{len(df)}")
        print(f"Samples with clarification_turn_question: {has_clar_turn_q}/{len(df)}")
        print(f"Samples with clarification_choice_text: {has_clar_choice_t}/{len(df)}")
        print(f"Samples marked as ambiguous: {is_ambiguous_count}/{len(df)}")
        
    except ImportError:
        print("❌ pandas/pyarrow not installed. Install with: pip install pandas pyarrow")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_data_fields.py <parquet_file_path> [num_samples]")
        sys.exit(1)
    
    parquet_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    check_parquet_data(parquet_path, num_samples)

