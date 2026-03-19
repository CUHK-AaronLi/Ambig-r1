#!/usr/bin/env python3
"""
Create mixed training datasets for Phase 1 experiments.

Combines multiple parquet datasets with proportional sampling to prevent
large datasets from dominating training.

Sampling strategies:
  - equal: Equal number of samples from each dataset
  - sqrt:  Sample proportional to sqrt(N) of each dataset size
  - full:  No sampling, concatenate all

Usage:
    # mix-B: PACIFIC + ShARC
    python scripts/create_mixed_dataset.py \
        --datasets pacific_fewshot sharc_fewshot \
        --output_dir scripts/data_process/data/mixed_b \
        --strategy sqrt

    # mix-C: PACIFIC + ShARC + SituatedQA
    python scripts/create_mixed_dataset.py \
        --datasets pacific_fewshot sharc_fewshot situatedqa_fewshot \
        --output_dir scripts/data_process/data/mixed_c \
        --strategy sqrt
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd


# Dataset directory mapping (relative to code/)
DATASET_DIRS = {
    'pacific_fewshot': 'scripts/data_process/data/pacific_fewshot',
    'abgcoqa': 'scripts/data_process/data/abgcoqa',
    'sharc_fewshot': 'scripts/data_process/data/sharc_fewshot',
    'situatedqa_fewshot': 'scripts/data_process/data/situatedqa_fewshot',
    'ambignq_fewshot': 'scripts/data_process/data/ambignq_fewshot',
}

# Short aliases
ALIASES = {
    'pacific': 'pacific_fewshot',
    'sharc': 'sharc_fewshot',
    'situatedqa': 'situatedqa_fewshot',
    'ambignq': 'ambignq_fewshot',
}


def resolve_name(name: str) -> str:
    """Resolve dataset alias to full name."""
    return ALIASES.get(name, name)


def load_dataset(base_dir: str, dataset_name: str) -> tuple:
    """Load train parquet for a dataset. Returns (df, name)."""
    full_name = resolve_name(dataset_name)
    data_dir = os.path.join(base_dir, DATASET_DIRS[full_name])
    train_path = os.path.join(data_dir, 'train.parquet')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    df = pd.read_parquet(train_path)
    print(f"  {full_name}: {len(df)} rows from {train_path}")
    return df, full_name


def compute_sample_sizes(sizes: dict, strategy: str) -> dict:
    """Compute how many samples to take from each dataset.

    Args:
        sizes: {dataset_name: total_rows}
        strategy: 'equal', 'sqrt', or 'full'

    Returns:
        {dataset_name: n_samples}
    """
    if strategy == 'full':
        return dict(sizes)

    if strategy == 'equal':
        min_size = min(sizes.values())
        return {k: min_size for k in sizes}

    if strategy == 'sqrt':
        # Sample proportional to sqrt(N), scaled so total ≈ sum of sqrt(N) * scaling_factor
        # Use the smallest dataset's full size as anchor
        sqrt_vals = {k: math.sqrt(v) for k, v in sizes.items()}
        total_sqrt = sum(sqrt_vals.values())

        # Scale so the largest contribution is capped at its actual size
        # and the smallest dataset contributes its full data
        min_name = min(sizes, key=sizes.get)
        min_size = sizes[min_name]
        min_sqrt = sqrt_vals[min_name]

        # scaling: min_dataset gets all its data, others proportional
        scale = min_size / min_sqrt

        result = {}
        for k in sizes:
            n = int(sqrt_vals[k] * scale)
            result[k] = min(n, sizes[k])  # never exceed actual size

        return result

    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description='Create mixed training dataset')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset names to combine (e.g., pacific_fewshot sharc_fewshot)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for mixed dataset')
    parser.add_argument('--strategy', type=str, default='sqrt',
                        choices=['equal', 'sqrt', 'full'],
                        help='Sampling strategy (default: sqrt)')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory (code/ on server)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    print(f"=== Creating Mixed Dataset ===")
    print(f"Datasets: {args.datasets}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output_dir}")
    print()

    # Load all datasets
    dfs = {}
    for name in args.datasets:
        df, full_name = load_dataset(args.base_dir, name)
        dfs[full_name] = df

    # Compute sample sizes
    sizes = {k: len(v) for k, v in dfs.items()}
    sample_sizes = compute_sample_sizes(sizes, args.strategy)

    print(f"\nSampling plan ({args.strategy}):")
    total_samples = 0
    for name in dfs:
        n = sample_sizes[name]
        total_samples += n
        pct = n / sizes[name] * 100
        print(f"  {name}: {n}/{sizes[name]} ({pct:.1f}%)")
    print(f"  Total: {total_samples}")

    # Sample and concatenate
    sampled = []
    for name, df in dfs.items():
        n = sample_sizes[name]
        if n >= len(df):
            sampled.append(df)
        else:
            sampled.append(df.sample(n=n, random_state=args.seed))

    mixed = pd.concat(sampled, ignore_index=True)

    # Shuffle
    mixed = mixed.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'train.parquet')
    mixed.to_parquet(out_path)

    # Save metadata
    meta_path = os.path.join(args.output_dir, 'mix_metadata.txt')
    with open(meta_path, 'w') as f:
        f.write(f"Mixed dataset created with strategy: {args.strategy}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Total: {len(mixed)}\n\n")
        for name in dfs:
            f.write(f"{name}: {sample_sizes[name]}/{sizes[name]}\n")

    print(f"\nSaved: {out_path} ({len(mixed)} rows)")
    print(f"Metadata: {meta_path}")


if __name__ == '__main__':
    main()
