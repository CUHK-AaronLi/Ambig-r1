#!/usr/bin/env python3
"""
Direction B: Analyze which examples benefit from clarification.
Uses existing cross-eval logs to understand per-example patterns.

Key question: What makes an example "clarification-helpful" vs "clarification-harmful"?
"""

import re
import pandas as pd
import os
from collections import defaultdict

LOG_DIR = 'scripts'

def parse_eval_log(log_path):
    """Parse a 5ds eval log to extract per-example trajectories."""
    if not os.path.exists(log_path):
        print(f'  SKIP: {log_path} not found')
        return []

    with open(log_path) as f:
        content = f.read()

    examples = []

    # Find all IPOv3 lines with F1 and clarification
    pattern = r'\[IPOv3\] F1=([\d.]+) turns=(\d+) .*?ACTION\s*:\s*\*\*([A-Z]+)\*\*\s*:?\s*(.*?)(?=\[IPOv3\]|$)'

    for m in re.finditer(pattern, content, re.DOTALL):
        f1 = float(m.group(1))
        turns = int(m.group(2))
        action = m.group(3)
        clar_text = m.group(4).strip()

        examples.append({
            'f1': f1,
            'turns': turns,
            'action': action,
            'clarify_text': clar_text[:100],
            'n_clarify': 1 if action == 'CLARIFY' else 0,
        })

    return examples


def analyze_clarification_helpfulness(examples):
    """Classify examples as clarification-helpful vs harmful."""

    clarify_examples = [e for e in examples if e['n_clarify'] > 0]
    direct_examples = [e for e in examples if e['n_clarify'] == 0]

    print(f'  Total: {len(examples)} | Clarify: {len(clarify_examples)} | Direct: {len(direct_examples)}')

    if clarify_examples and direct_examples:
        avg_clarify_f1 = sum(e['f1'] for e in clarify_examples) / len(clarify_examples)
        avg_direct_f1 = sum(e['f1'] for e in direct_examples) / len(direct_examples)
        print(f'  Avg F1 (clarify): {avg_clarify_f1:.3f} | Avg F1 (direct): {avg_direct_f1:.3f}')

        # Find examples where clarification HURT (clarify F1 < direct F1 baseline)
        # Since we don't have paired data, this is just aggregate

    return clarify_examples, direct_examples


def analyze_by_turns(examples):
    """Analyze F1 by number of clarification turns."""
    by_turns = defaultdict(list)
    for e in examples:
        by_turns[e['turns']].append(e['f1'])

    print('\n  F1 by clarification turns:')
    for turns in sorted(by_turns.keys()):
        f1s = by_turns[turns]
        print(f'    Turns={turns}: n={len(f1s)}, mean_F1={sum(f1s)/len(f1s):.3f}')


def main():
    print('='*70)
    print('Direction B: Clarification Helpfulness Analysis')
    print('='*70)

    # Analyze SH log files
    log_files = {
        'sharc': f'{LOG_DIR}/5ds-xeval-sharc_fewshot.log',
        'situatedqa': f'{LOG_DIR}/5ds-xeval-situatedqa_fewshot.log',
    }

    for ds, log_path in log_files.items():
        if not os.path.exists(log_path):
            print(f'\n--- {ds} ---')
            print(f'  SKIP: {log_path} not found')
            continue

        print(f'\n--- {ds} ---')
        examples = parse_eval_log(log_path)
        clarify_ex, direct_ex = analyze_clarification_helpfulness(examples)
        analyze_by_turns(examples)

        if clarify_ex:
            # Show some examples
            print('\n  Sample clarification examples:')
            for e in clarify_ex[:3]:
                print(f'    F1={e["f1"]:.1f} turns={e["turns"]}: {e["clarify_text"][:80]}...')

    # Analyze by data source in mix_5ds if available
    print('\n' + '='*70)
    print('Per-dataset analysis from mix_5ds training data:')
    print('='*70)

    mix_path = f'{LOG_DIR}/data_process/data/mix_5ds/train.parquet'
    if os.path.exists(mix_path):
        df = pd.read_parquet(mix_path)
        print(f'\n  mix_5ds: {len(df)} rows')
        print(f'  Sources: {df["data_source"].value_counts().to_dict()}')

        # Sample some prompts from each dataset
        for ds_name in df['data_source'].unique():
            df_ds = df[df['data_source'] == ds_name]
            print(f'\n  --- {ds_name} ({len(df_ds)} examples) ---')

            # Show a sample prompt
            row = df_ds.iloc[0]
            prompt = row.get('prompt', '')
            if isinstance(prompt, list):
                user_msgs = [m['content'][:150] for m in prompt if m.get('role') == 'user']
                if user_msgs:
                    print(f'    Sample: {user_msgs[0][:120]}...')

    print('\n' + '='*70)
    print('FILTER DESIGN RECOMMENDATION:')
    print('='*70)
    print("""
Based on available data:
- PACIFIC: Keep all (clarification generally helps)
- SHARC: Keep all (clarification helps on average)
- AbgCoQA: Filter by example difficulty (easy questions don't need clarification)
- AmbigNQ: Filter by question complexity (simple entity Qs don't need clarification)
- SituatedQA: Keep selective (clarification often hurts)

Proposed filter: post_clarify_f1 > no_clarify_f1 threshold
→ But this requires running eval on training data first
→ Recommend: Use mix_5ds as-is, trust RL to learn the filter
""")


if __name__ == '__main__':
    main()
