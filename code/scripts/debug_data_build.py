#!/usr/bin/env python3
"""
调试数据构建过程，检查 clarification 相关字段
"""
import json
import sys
from pathlib import Path

def check_raw_json(json_path: str, num_samples: int = 3):
    """检查原始 JSON 文件中的数据"""
    print(f"Reading JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTotal examples: {len(data)}")
    print(f"{'='*80}")
    
    for idx in range(min(num_samples, len(data))):
        example = data[idx]
        print(f"\n--- Example {idx} (id: {example.get('id', 'N/A')}) ---")
        
        # 检查关键字段
        target_turn = example.get('target_turn', {})
        clarification_turn = example.get('clarification_turn', {})
        clarification_answers = clarification_turn.get('answers', [])
        ambiguity = example.get('ambiguity', '')
        
        print(f"ambiguity: '{ambiguity}'")
        print(f"target_turn.question: '{target_turn.get('question', 'N/A')[:100]}'")
        print(f"clarification_turn: {clarification_turn}")
        print(f"clarification_turn.question: '{clarification_turn.get('question', 'N/A')[:100] if clarification_turn.get('question') else 'EMPTY'}'")
        print(f"clarification_turn.answers count: {len(clarification_answers)}")
        
        if clarification_answers:
            print(f"\nFirst clarification answer:")
            first_ans = clarification_answers[0]
            print(f"  Keys: {list(first_ans.keys())}")
            print(f"  clr_ans: '{first_ans.get('clr_ans', 'N/A')[:100] if first_ans.get('clr_ans') else 'EMPTY'}'")
            org_ans_keys = [k for k in first_ans.keys() if k.startswith('org_ans')]
            print(f"  org_ans keys: {org_ans_keys}")
            if org_ans_keys:
                print(f"  First org_ans value: '{first_ans.get(org_ans_keys[0], 'N/A')[:100] if first_ans.get(org_ans_keys[0]) else 'EMPTY'}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_data_build.py <json_file_path> [num_samples]")
        print("\nExample:")
        print("  python debug_data_build.py data/coqa_abg_test.json 5")
        sys.exit(1)
    
    json_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    check_raw_json(json_path, num_samples)

