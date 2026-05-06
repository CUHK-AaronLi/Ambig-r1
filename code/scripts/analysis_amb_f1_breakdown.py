"""A9: Per-Ambiguity F1 Breakdown — the KEY analysis for the paper.
For each val sample: generate response → check clarify → compute F1 → cross with ambiguity label.
Produces 2×2: F1(amb+clr), F1(amb+no_clr), F1(nonamb+clr), F1(nonamb+no_clr).
"""
import pandas as pd
import numpy as np
import re
import json
import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def compute_token_f1(prediction, reference):
    """Token-level F1 (same as reward_utils)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    return 2 * p * r / (p + r)

def max_f1(prediction, references):
    """Max F1 across multiple gold references."""
    if not references:
        return 0.0
    return max(compute_token_f1(prediction, ref) for ref in references)

def extract_answer(text):
    """Extract answer from model response (same logic as reward_utils)."""
    # Try <answer> tag
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.I)
    if m:
        return m.group(1).strip()
    # Fallback: text after last </user_response>
    if '</user_response>' in text:
        last_ur = text.rfind('</user_response>')
        after = text[last_ur + len('</user_response>'):].strip()
        if after:
            after = re.sub(r'^(?:answer|Answer|ANSWER)\s*[:：]\s*', '', after).strip()
            return after
    return text.strip()

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "verl_checkpoints_diag/E12c-nb/actor/global_step_250"
    data_path = "scripts/data_process/data/mix_5ds/val_pacific.parquet"

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    llm = LLM(model=ckpt, trust_remote_code=True, dtype="float16",
              max_model_len=6144, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=512)

    results = []
    for i in range(len(df)):
        row = df.iloc[i]

        # Get ambiguity label
        rm = row.get("reward_model", {})
        if hasattr(rm, "tolist"): rm = rm.tolist()
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt = rm.get("ground_truth", {}) if isinstance(rm, dict) else {}
        if not isinstance(gt, dict): gt = {}

        is_amb = gt.get("req_clari", False)

        # Also check extra_info for _is_ambiguous
        ei = row.get("extra_info", {})
        if hasattr(ei, "tolist"): ei = ei.tolist()
        if isinstance(ei, str):
            try: ei = json.loads(ei)
            except: ei = {}
        if isinstance(ei, dict) and ei.get("_is_ambiguous", False):
            is_amb = True

        # Get gold answers
        target = gt.get("target", [])
        if hasattr(target, "tolist"): target = target.tolist()
        if isinstance(target, str): target = [target]
        if not isinstance(target, list): target = [str(target)]
        gold_answers = [str(t).strip() for t in target if str(t).strip()]

        # Generate response
        p = row["prompt"]
        if hasattr(p, "tolist"): p = p.tolist()
        prompt_text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        output = llm.generate([prompt_text], params)
        response = output[0].outputs[0].text.strip()

        # Detect clarify
        has_clarify = bool(re.search(r'<clarify>', response, re.I))
        has_answer = bool(re.search(r'<answer>', response, re.I))

        # Extract answer and compute F1
        answer_text = extract_answer(response)
        f1 = max_f1(answer_text, gold_answers) if gold_answers else 0.0

        # Extract clarify question
        clr_match = re.search(r'<clarify>(.*?)</clarify>', response, re.DOTALL | re.I)
        clr_q = clr_match.group(1).strip() if clr_match else ""

        results.append({
            "idx": i,
            "is_ambiguous": bool(is_amb),
            "has_clarify": has_clarify,
            "has_answer": has_answer,
            "f1": f1,
            "answer_text": answer_text[:200],
            "clarify_question": clr_q,
            "response_len": len(response),
            "n_gold": len(gold_answers),
        })

        if (i + 1) % 100 == 0:
            _print_running_stats(results, i + 1, len(df))

    _print_final_stats(results, ckpt)

def _print_running_stats(results, done, total):
    amb_clr = [r for r in results if r["is_ambiguous"] and r["has_clarify"]]
    amb_noclr = [r for r in results if r["is_ambiguous"] and not r["has_clarify"]]
    nonamb_clr = [r for r in results if not r["is_ambiguous"] and r["has_clarify"]]
    nonamb_noclr = [r for r in results if not r["is_ambiguous"] and not r["has_clarify"]]

    f1_ac = np.mean([r["f1"] for r in amb_clr]) if amb_clr else 0
    f1_an = np.mean([r["f1"] for r in amb_noclr]) if amb_noclr else 0
    f1_nc = np.mean([r["f1"] for r in nonamb_clr]) if nonamb_clr else 0
    f1_nn = np.mean([r["f1"] for r in nonamb_noclr]) if nonamb_noclr else 0
    f1_all = np.mean([r["f1"] for r in results])

    print(f"[{done}/{total}] F1={f1_all:.3f} | "
          f"amb+clr={f1_ac:.3f}(n={len(amb_clr)}) "
          f"amb+no={f1_an:.3f}(n={len(amb_noclr)}) "
          f"nonamb+clr={f1_nc:.3f}(n={len(nonamb_clr)}) "
          f"nonamb+no={f1_nn:.3f}(n={len(nonamb_noclr)})")

def _print_final_stats(results, ckpt):
    amb_clr = [r for r in results if r["is_ambiguous"] and r["has_clarify"]]
    amb_noclr = [r for r in results if r["is_ambiguous"] and not r["has_clarify"]]
    nonamb_clr = [r for r in results if not r["is_ambiguous"] and r["has_clarify"]]
    nonamb_noclr = [r for r in results if not r["is_ambiguous"] and not r["has_clarify"]]

    f1_ac = np.mean([r["f1"] for r in amb_clr]) if amb_clr else 0
    f1_an = np.mean([r["f1"] for r in amb_noclr]) if amb_noclr else 0
    f1_nc = np.mean([r["f1"] for r in nonamb_clr]) if nonamb_clr else 0
    f1_nn = np.mean([r["f1"] for r in nonamb_noclr]) if nonamb_noclr else 0
    f1_all = np.mean([r["f1"] for r in results])

    print(f"\n{'='*70}")
    print(f"Per-Ambiguity F1 Breakdown ({ckpt})")
    print(f"{'='*70}")
    print(f"Total: {len(results)}, Overall F1: {f1_all:.3f}")
    print(f"\n{'':>20}{'Ambiguous':>15}{'Non-ambiguous':>15}")
    print(f"{'Clarified':>20}  F1={f1_ac:.3f} (n={len(amb_clr):>3})   F1={f1_nc:.3f} (n={len(nonamb_clr):>3})")
    print(f"{'Not clarified':>20}  F1={f1_an:.3f} (n={len(amb_noclr):>3})   F1={f1_nn:.3f} (n={len(nonamb_noclr):>3})")
    print(f"\nKey comparisons:")
    delta_amb = f1_ac - f1_an
    print(f"  Clarify helps on ambiguous? F1(amb+clr) - F1(amb+no_clr) = {delta_amb:+.3f}")
    delta_nonamb = f1_nc - f1_nn
    print(f"  Clarify hurts non-ambiguous? F1(nonamb+clr) - F1(nonamb+no_clr) = {delta_nonamb:+.3f}")
    print(f"  Decision precision: {len(amb_clr)}/{len(amb_clr)+len(nonamb_clr)} = {len(amb_clr)/(len(amb_clr)+len(nonamb_clr)):.3f}" if (len(amb_clr)+len(nonamb_clr)) > 0 else "")
    print(f"{'='*70}")

    # Save
    out_path = f"eval_results/amb_f1_breakdown_{ckpt.split('/')[-1]}.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "f1_overall": f1_all,
                "f1_amb_clr": f1_ac, "n_amb_clr": len(amb_clr),
                "f1_amb_noclr": f1_an, "n_amb_noclr": len(amb_noclr),
                "f1_nonamb_clr": f1_nc, "n_nonamb_clr": len(nonamb_clr),
                "f1_nonamb_noclr": f1_nn, "n_nonamb_noclr": len(nonamb_noclr),
            },
            "results": results
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
