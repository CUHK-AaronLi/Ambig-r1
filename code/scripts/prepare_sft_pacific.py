"""
Prepare combined SFT data: existing clarify SFT data + PACIFIC direct-answer examples.

Loads:
  - Existing SFT clarify data (prompt/response string columns)
  - PACIFIC fewshot train data (RL format with reward_model dict)

For each PACIFIC example, creates a direct-answer SFT pair using the golden answers.
Samples 3000 PACIFIC examples, merges with existing SFT data, splits 95/5 train/val.

Output: data_process/data/sft_pacific_7b/{train,val}.parquet
"""

import os
import json
import random
import argparse

import pandas as pd


def extract_pacific_sft_examples(pacific_df: pd.DataFrame, n_samples: int = 3000, seed: int = 42) -> list:
    """Convert PACIFIC RL-format examples to SFT (prompt, response) pairs."""
    random.seed(seed)
    examples = []

    for idx, row in pacific_df.iterrows():
        # Extract prompt text from the prompt column
        # PACIFIC prompt is stored as list: [{"role": "user", "content": "..."}]
        prompt_data = row["prompt"]
        # prompt may be numpy.ndarray, list, or str
        if hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()  # numpy array -> list
        if isinstance(prompt_data, list):
            # Standard format: [{"role": "user", "content": "..."}]
            prompt_text = prompt_data[0]["content"] if len(prompt_data) > 0 else ""
        elif isinstance(prompt_data, str):
            prompt_text = prompt_data
        else:
            continue

        if not prompt_text.strip():
            continue

        # Extract golden answers from reward_model
        reward_model = row["reward_model"]
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(reward_model, dict):
            continue

        ground_truth = reward_model.get("ground_truth", {})
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except (json.JSONDecodeError, TypeError):
                ground_truth = {"target": [ground_truth]}

        target = ground_truth.get("target", [])
        if hasattr(target, 'tolist'):
            target = target.tolist()  # numpy array -> list
        if not target:
            continue

        # Use the first answer
        golden_answer = target[0] if isinstance(target, list) else str(target)
        golden_answer = str(golden_answer).strip()
        if not golden_answer:
            continue

        # Build the SFT response: direct answer with think tag
        response = (
            f"<think>I can answer this question directly based on the provided data.</think>\n"
            f"<answer>{golden_answer}</answer>"
        )

        examples.append({
            "prompt": prompt_text,
            "response": response,
        })

    print(f"Total valid PACIFIC examples: {len(examples)}")

    # Sample n_samples
    if len(examples) > n_samples:
        random.shuffle(examples)
        examples = examples[:n_samples]
        print(f"Sampled {n_samples} PACIFIC examples")
    else:
        print(f"Using all {len(examples)} PACIFIC examples (fewer than {n_samples})")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare combined SFT data with PACIFIC examples")
    parser.add_argument("--sft_data", type=str,
                        default="data_process/data/sft_clarify/train.parquet",
                        help="Path to existing SFT train parquet")
    parser.add_argument("--pacific_data", type=str,
                        default="data_process/data/pacific_fewshot/train.parquet",
                        help="Path to PACIFIC fewshot train parquet")
    parser.add_argument("--output_dir", type=str,
                        default="data_process/data/sft_pacific_7b",
                        help="Output directory for combined SFT data")
    parser.add_argument("--n_pacific", type=int, default=3000,
                        help="Number of PACIFIC examples to sample")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load existing SFT data ---
    print(f"Loading existing SFT data from {args.sft_data}")
    sft_df = pd.read_parquet(args.sft_data)
    print(f"  Existing SFT rows: {len(sft_df)}")
    print(f"  Columns: {sft_df.columns.tolist()}")

    existing_examples = []
    for _, row in sft_df.iterrows():
        existing_examples.append({
            "prompt": str(row["prompt"]),
            "response": str(row["response"]),
        })

    # Count existing PACIFIC examples (check if prompt mentions "Context:" which is PACIFIC-specific)
    pacific_in_existing = sum(1 for ex in existing_examples if "Context:" in ex["prompt"][:500])
    print(f"  PACIFIC-like examples in existing SFT: {pacific_in_existing}")

    # --- Load PACIFIC data ---
    print(f"\nLoading PACIFIC data from {args.pacific_data}")
    pacific_df = pd.read_parquet(args.pacific_data)
    print(f"  PACIFIC rows: {len(pacific_df)}")
    print(f"  Columns: {pacific_df.columns.tolist()}")

    pacific_examples = extract_pacific_sft_examples(pacific_df, n_samples=args.n_pacific, seed=args.seed)

    # Filter out too-long examples (>3500 chars to stay under 4096 tokens)
    before = len(pacific_examples)
    pacific_examples = [ex for ex in pacific_examples if len(ex["prompt"]) + len(ex["response"]) < 3500]
    print(f"Filtered PACIFIC: {before} -> {len(pacific_examples)} (removed {before - len(pacific_examples)} too-long)")

    # --- Merge ---
    all_examples = existing_examples + pacific_examples
    random.shuffle(all_examples)

    print(f"\nCombined dataset: {len(all_examples)} examples")
    print(f"  From existing SFT: {len(existing_examples)}")
    print(f"  From PACIFIC: {len(pacific_examples)}")

    # --- Split train/val ---
    n_val = max(1, int(len(all_examples) * args.val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    print(f"\nTrain: {len(train_examples)}, Val: {len(val_examples)}")

    # --- Analyze composition ---
    n_clarify_train = sum(1 for ex in train_examples if "<clarify>" in ex["response"])
    n_direct_train = sum(1 for ex in train_examples if "<answer>" in ex["response"] and "<clarify>" not in ex["response"])
    n_pacific_train = sum(1 for ex in train_examples if "Context:" in ex["prompt"][:500])
    print(f"\nTrain composition:")
    print(f"  Clarify responses: {n_clarify_train}")
    print(f"  Direct answer responses: {n_direct_train}")
    print(f"  PACIFIC-origin prompts: {n_pacific_train}")

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_examples)
    val_df = pd.DataFrame(val_examples)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved train to {train_path} ({len(train_df)} rows)")
    print(f"Saved val to {val_path} ({len(val_df)} rows)")

    # --- Print samples ---
    print(f"\n{'='*60}")
    print("Sample PACIFIC SFT example:")
    print(f"{'='*60}")
    if pacific_examples:
        sample = pacific_examples[0]
        print(f"Prompt (first 300 chars):\n{sample['prompt'][:300]}...")
        print(f"\nResponse:\n{sample['response']}")
    print(f"{'='*60}")

    print("\nDone.")


if __name__ == "__main__":
    main()
