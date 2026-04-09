#!/usr/bin/env python3
"""Create mixed training dataset for Stage 2 training.
Samples N examples from each dataset and concatenates.
"""

import pandas as pd
import os

DATA_DIR = 'scripts/data_process/data'
OUTPUT_DIR = 'scripts/data_process/data/mix_5ds'

# How many samples to take from each dataset
N_PER_DS = 500  # balanced sampling

DATASETS = {
    'pacific_fewshot': 'pacific',
    'abgcoqa': 'abgcoqa',
    'ambignq_fewshot': 'ambignq',
    'sharc_fewshot': 'sharc',
    'situatedqa_fewshot': 'situatedqa',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

frames = []
for ds_dir, ds_name in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_dir, 'train.parquet')
    if not os.path.exists(path):
        print(f'  SKIP {ds_dir}: no train.parquet')
        continue

    df = pd.read_parquet(path)
    n = min(N_PER_DS, len(df))
    df_sample = df.sample(n=n, random_state=42)
    frames.append(df_sample)
    print(f'  {ds_dir}: {len(df)} total, sampled {n}')

# Combine and shuffle
combined = pd.concat(frames, ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
train_path = os.path.join(OUTPUT_DIR, 'train.parquet')
combined.to_parquet(train_path, index=False)
print(f'\nSaved {len(combined)} examples to {train_path}')
print(f'Data sources: {combined["data_source"].value_counts().to_dict()}')

# Also save individual val files for per-dataset evaluation
for ds_dir, ds_name in DATASETS.items():
    for val_name in ['validation.parquet', 'test.parquet']:
        src = os.path.join(DATA_DIR, ds_dir, val_name)
        if os.path.exists(src):
            dst = os.path.join(OUTPUT_DIR, f'val_{ds_name}.parquet')
            df_val = pd.read_parquet(src)
            df_val.to_parquet(dst, index=False)
            print(f'  Val: {ds_name} ({len(df_val)} rows)')
            break
