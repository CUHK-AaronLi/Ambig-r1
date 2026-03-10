#!/usr/bin/env python3
"""
Check AmbigNQ val set (tinytest.parquet) quality.

Validates:
1. golden_answers format correctness
2. No empty answers
3. Ambiguous vs non-ambiguous balance
4. Question length distribution

Usage:
    python scripts/check_val_data.py scripts/data_process/data/ambignq_fewshot/tinytest.parquet
"""

import sys
import json
import ast

import pandas as pd
import numpy as np


def check_val_data(path: str):
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} samples from {path}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Inspect first row to understand structure
    row0 = df.iloc[0]
    print("=== First row sample ===")
    for col in df.columns:
        val = row0[col]
        if isinstance(val, (list, dict)):
            print(f"  {col}: {type(val).__name__} = {str(val)[:200]}")
        elif isinstance(val, np.ndarray):
            print(f"  {col}: ndarray shape={val.shape}")
        else:
            print(f"  {col}: {type(val).__name__} = {str(val)[:200]}")
    print()

    # Extract ground truth info
    empty_gt = 0
    gt_counts = []  # number of reference answers per sample
    has_ambig_label = 0
    ambig_count = 0
    non_ambig_count = 0
    question_lengths = []

    for idx, row in df.iterrows():
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, str):
            try:
                extra_info = ast.literal_eval(extra_info)
            except Exception:
                extra_info = {}

        # Ground truth — check both top-level reward_model column and extra_info
        gt = None
        rm = row.get('reward_model', None)
        if isinstance(rm, dict):
            gt = rm.get('ground_truth', None)
        if gt is None:
            rm2 = extra_info.get('reward_model', {})
            if isinstance(rm2, dict):
                gt = rm2.get('ground_truth', None)

        if gt is None:
            empty_gt += 1
            gt_counts.append(0)
        else:
            refs = []
            if isinstance(gt, dict):
                for k in ('target', 'answers'):
                    v = gt.get(k)
                    if v is not None:
                        refs.extend(v if isinstance(v, (list, tuple)) else [v])
            elif isinstance(gt, (list, tuple)):
                refs = list(gt)
            else:
                refs = [gt]
            refs = [str(r).strip() for r in refs if str(r).strip()]
            gt_counts.append(len(refs))
            if len(refs) == 0:
                empty_gt += 1

        # Ambiguity label (check top-level column first, then extra_info)
        ambig = row.get('_is_ambiguous', extra_info.get('_is_ambiguous',
                extra_info.get('is_ambiguous', extra_info.get('ambiguous', None))))
        if ambig is not None:
            has_ambig_label += 1
            if ambig:
                ambig_count += 1
            else:
                non_ambig_count += 1

        # Question (top-level column preferred)
        q = row.get('question', extra_info.get('question', ''))
        if q:
            question_lengths.append(len(q))

    # Report
    print("=== Ground Truth Quality ===")
    print(f"  Total samples:     {len(df)}")
    print(f"  Empty GT:          {empty_gt} ({100*empty_gt/len(df):.1f}%)")
    gt_arr = np.array(gt_counts)
    print(f"  Refs per sample:   mean={gt_arr.mean():.1f}, min={gt_arr.min()}, max={gt_arr.max()}")
    print(f"  Single-ref:        {(gt_arr == 1).sum()} ({100*(gt_arr == 1).sum()/len(df):.1f}%)")
    print(f"  Multi-ref (>1):    {(gt_arr > 1).sum()} ({100*(gt_arr > 1).sum()/len(df):.1f}%)")
    print()

    print("=== Ambiguity Labels ===")
    if has_ambig_label > 0:
        print(f"  Labeled samples:   {has_ambig_label}")
        print(f"  Ambiguous:         {ambig_count} ({100*ambig_count/has_ambig_label:.1f}%)")
        print(f"  Non-ambiguous:     {non_ambig_count} ({100*non_ambig_count/has_ambig_label:.1f}%)")
    else:
        # Heuristic: multi-ref often means ambiguous in AmbigNQ
        multi_ref = (gt_arr > 1).sum()
        print(f"  No explicit labels found.")
        print(f"  Heuristic (multi-ref = ambiguous): {multi_ref}/{len(df)} ({100*multi_ref/len(df):.1f}%)")
    print()

    if question_lengths:
        ql = np.array(question_lengths)
        print("=== Question Length ===")
        print(f"  Mean: {ql.mean():.0f} chars, Median: {np.median(ql):.0f}, Max: {ql.max()}")
    print()

    # Spot check: print 5 samples
    print("=== Sample Spot Check (first 5) ===")
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, str):
            try:
                extra_info = ast.literal_eval(extra_info)
            except Exception:
                extra_info = {}
        q = str(row.get('question', ''))[:100]
        rm = row.get('reward_model', None)
        gt = rm.get('ground_truth', None) if isinstance(rm, dict) else None
        gt_str = str(gt)[:100] if gt else "EMPTY"
        print(f"  [{idx}] Q: {q}")
        print(f"       GT: {gt_str}")
    print()

    # Summary
    issues = []
    if empty_gt > 0:
        issues.append(f"{empty_gt} samples have empty ground truth")
    if gt_arr.min() == 0 and empty_gt > 0:
        issues.append("Some samples will always get F1=0 (no references)")

    if issues:
        print("=== ISSUES FOUND ===")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("=== NO ISSUES FOUND ===")
        print("  Val data looks clean.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_val_data.py <parquet_path>")
        sys.exit(1)
    check_val_data(sys.argv[1])
