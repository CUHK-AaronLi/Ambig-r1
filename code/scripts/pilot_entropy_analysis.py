"""
Pilot: Measure model uncertainty (entropy) on PACIFIC validation set.
For each question, sample N completions and measure:
1. Mean logprob of generated answers (lower = more uncertain)
2. Answer agreement across N samples (lower = more uncertain)
3. Split by req_clari to see if ambiguous questions have higher uncertainty

This helps set the threshold for EVPI-gated reward.
"""
import argparse
import json
import numpy as np
import pyarrow.parquet as pq
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_completions", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    # Load validation data
    print(f"Loading val data from {args.val_data}")
    table = pq.read_table(args.val_data)
    n_total = len(table)

    # Sample n_samples
    np.random.seed(42)
    indices = np.random.choice(n_total, min(args.n_samples, n_total), replace=False)

    # Extract prompts and labels
    prompts_col = table.column("prompt").to_pylist()
    extra_infos = table.column("extra_info").to_pylist()
    reward_models = table.column("reward_model").to_pylist()

    samples = []
    for idx in indices:
        prompt_data = prompts_col[idx]
        if hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            prompt_text = prompt_data[0].get("content", "")
        else:
            prompt_text = str(prompt_data)

        ei = extra_infos[idx]
        req_clari = ei.get("req_clari", False) if isinstance(ei, dict) else False

        rm = reward_models[idx]
        gt = rm.get("ground_truth", {}) if isinstance(rm, dict) else {}
        target = gt.get("target", [])
        if hasattr(target, 'tolist'):
            target = target.tolist()
        golden = target[0] if isinstance(target, list) and len(target) > 0 else ""

        samples.append({
            "idx": int(idx),
            "prompt": prompt_text,
            "req_clari": bool(req_clari),
            "golden_answer": str(golden),
        })

    n_ambig = sum(1 for s in samples if s["req_clari"])
    print(f"Selected {len(samples)} samples: {n_ambig} ambiguous, {len(samples)-n_ambig} non-ambiguous")

    # Initialize vLLM
    print(f"\nLoading model from {args.model_path}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_completions,
        logprobs=1,  # return logprobs for generated tokens
    )

    # Generate completions
    print(f"\nGenerating {args.n_completions} completions per sample...")
    prompts = [s["prompt"] for s in samples]
    outputs = llm.generate(prompts, sampling_params)

    # Analyze entropy
    print(f"\n{'='*70}")
    print("ENTROPY ANALYSIS")
    print(f"{'='*70}")

    results_ambig = []
    results_non_ambig = []

    for i, (sample, output) in enumerate(zip(samples, outputs)):
        # Collect generated answers
        answers = []
        mean_logprobs = []

        for completion in output.outputs:
            text = completion.text.strip()
            # Extract answer from <answer>...</answer> if present
            import re
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            answer = match.group(1).strip() if match else text[:100]
            answers.append(answer)

            # Mean logprob
            if completion.logprobs:
                token_logprobs = [lp.logprob for step in completion.logprobs for token_id, lp in step.items()]
                if token_logprobs:
                    mean_logprobs.append(np.mean(token_logprobs))

        # Answer agreement: what fraction of completions give the same answer?
        if answers:
            most_common = Counter(answers).most_common(1)[0]
            agreement = most_common[1] / len(answers)
        else:
            agreement = 0.0

        # Mean logprob across completions
        avg_logprob = np.mean(mean_logprobs) if mean_logprobs else 0.0

        # Entropy proxy: -avg_logprob (higher = more uncertain)
        entropy_proxy = -avg_logprob

        result = {
            "idx": sample["idx"],
            "req_clari": sample["req_clari"],
            "agreement": agreement,
            "avg_logprob": avg_logprob,
            "entropy_proxy": entropy_proxy,
            "n_unique_answers": len(set(answers)),
            "golden": sample["golden_answer"],
            "sample_answer": answers[0] if answers else "",
        }

        if sample["req_clari"]:
            results_ambig.append(result)
        else:
            results_non_ambig.append(result)

    # Print statistics
    def print_stats(name, results):
        if not results:
            print(f"\n{name}: no samples")
            return
        agreements = [r["agreement"] for r in results]
        entropies = [r["entropy_proxy"] for r in results]
        n_unique = [r["n_unique_answers"] for r in results]
        print(f"\n{name} (n={len(results)}):")
        print(f"  Agreement:     mean={np.mean(agreements):.3f}, std={np.std(agreements):.3f}, min={np.min(agreements):.3f}, max={np.max(agreements):.3f}")
        print(f"  Entropy proxy: mean={np.mean(entropies):.3f}, std={np.std(entropies):.3f}, min={np.min(entropies):.3f}, max={np.max(entropies):.3f}")
        print(f"  Unique answers: mean={np.mean(n_unique):.1f}, median={np.median(n_unique):.0f}")

    print_stats("AMBIGUOUS (req_clari=True)", results_ambig)
    print_stats("NON-AMBIGUOUS (req_clari=False)", results_non_ambig)

    # Separation analysis
    if results_ambig and results_non_ambig:
        ambig_entropy = np.mean([r["entropy_proxy"] for r in results_ambig])
        non_ambig_entropy = np.mean([r["entropy_proxy"] for r in results_non_ambig])
        ambig_agreement = np.mean([r["agreement"] for r in results_ambig])
        non_ambig_agreement = np.mean([r["agreement"] for r in results_non_ambig])

        print(f"\n{'='*70}")
        print("SEPARATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Entropy gap:    ambig={ambig_entropy:.3f} vs non_ambig={non_ambig_entropy:.3f} (diff={ambig_entropy-non_ambig_entropy:.3f})")
        print(f"Agreement gap:  ambig={ambig_agreement:.3f} vs non_ambig={non_ambig_agreement:.3f} (diff={non_ambig_agreement-ambig_agreement:.3f})")

        if ambig_entropy > non_ambig_entropy:
            print("\n✓ Ambiguous questions have HIGHER entropy → entropy can be used as gating signal!")
            # Suggest threshold
            all_entropies = [r["entropy_proxy"] for r in results_ambig + results_non_ambig]
            p25 = np.percentile(all_entropies, 25)
            p50 = np.percentile(all_entropies, 50)
            p75 = np.percentile(all_entropies, 75)
            print(f"  Suggested thresholds: p25={p25:.3f}, p50={p50:.3f}, p75={p75:.3f}")
        else:
            print("\n✗ No clear entropy separation → entropy alone may not work as gating signal")

    # Print some examples
    print(f"\n{'='*70}")
    print("EXAMPLES")
    print(f"{'='*70}")

    all_results = sorted(results_ambig + results_non_ambig, key=lambda r: r["entropy_proxy"], reverse=True)

    print("\nTop 3 HIGHEST entropy (most uncertain):")
    for r in all_results[:3]:
        print(f"  req_clari={r['req_clari']}, entropy={r['entropy_proxy']:.3f}, agreement={r['agreement']:.1f}, unique={r['n_unique_answers']}")
        print(f"    golden: {r['golden'][:80]}")
        print(f"    model:  {r['sample_answer'][:80]}")

    print("\nTop 3 LOWEST entropy (most confident):")
    for r in all_results[-3:]:
        print(f"  req_clari={r['req_clari']}, entropy={r['entropy_proxy']:.3f}, agreement={r['agreement']:.1f}, unique={r['n_unique_answers']}")
        print(f"    golden: {r['golden'][:80]}")
        print(f"    model:  {r['sample_answer'][:80]}")


if __name__ == "__main__":
    main()
