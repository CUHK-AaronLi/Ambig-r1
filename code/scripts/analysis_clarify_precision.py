"""A1: Clarification Decision Quality — precision/recall on ambiguous questions.
Run as val-only with E12c checkpoint, output per-sample amb + clarify labels."""
import pandas as pd
import numpy as np
import re
import json
import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "verl_checkpoints_diag/E12c-nb/actor/global_step_125"
    data_path = "scripts/data_process/data/mix_5ds/val_pacific.parquet"

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    llm = LLM(model=ckpt, trust_remote_code=True, dtype="float16",
              max_model_len=6144, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=512)

    results = []
    for i in range(len(df)):
        ei = df.iloc[i].get("extra_info", {})
        if not isinstance(ei, dict):
            ei = {}

        is_amb = ei.get("_is_ambiguous", False)
        if not is_amb:
            # PACIFIC uses req_clari
            rm = df.iloc[i].get("reward_model", {})
            if isinstance(rm, dict):
                gt = rm.get("ground_truth", {})
                if isinstance(gt, dict):
                    is_amb = gt.get("req_clari", False)

        p = df.iloc[i]["prompt"]
        if hasattr(p, "tolist"):
            p = p.tolist()
        prompt_text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)

        output = llm.generate([prompt_text], params)
        response = output[0].outputs[0].text.strip()

        has_clarify = bool(re.search(r'<clarify>', response, re.I))
        has_answer = bool(re.search(r'<answer>', response, re.I))

        # Extract clarify question if any
        clr_match = re.search(r'<clarify>(.*?)</clarify>', response, re.DOTALL | re.I)
        clr_question = clr_match.group(1).strip() if clr_match else ""

        results.append({
            "idx": i,
            "is_ambiguous": bool(is_amb),
            "has_clarify": has_clarify,
            "has_answer": has_answer,
            "clarify_question": clr_question,
            "response_len": len(response),
        })

        if (i + 1) % 100 == 0:
            # Running stats
            amb_clr = sum(1 for r in results if r["is_ambiguous"] and r["has_clarify"])
            amb_noclr = sum(1 for r in results if r["is_ambiguous"] and not r["has_clarify"])
            nonamb_clr = sum(1 for r in results if not r["is_ambiguous"] and r["has_clarify"])
            nonamb_noclr = sum(1 for r in results if not r["is_ambiguous"] and not r["has_clarify"])
            total_clr = amb_clr + nonamb_clr
            precision = amb_clr / total_clr if total_clr > 0 else 0
            recall = amb_clr / (amb_clr + amb_noclr) if (amb_clr + amb_noclr) > 0 else 0
            print(f"[{i+1}/{len(df)}] Precision={precision:.3f} Recall={recall:.3f} "
                  f"amb_clr={amb_clr} amb_noclr={amb_noclr} nonamb_clr={nonamb_clr} nonamb_noclr={nonamb_noclr}")

    # Final stats
    amb_clr = sum(1 for r in results if r["is_ambiguous"] and r["has_clarify"])
    amb_noclr = sum(1 for r in results if r["is_ambiguous"] and not r["has_clarify"])
    nonamb_clr = sum(1 for r in results if not r["is_ambiguous"] and r["has_clarify"])
    nonamb_noclr = sum(1 for r in results if not r["is_ambiguous"] and not r["has_clarify"])
    total_clr = amb_clr + nonamb_clr
    total_amb = amb_clr + amb_noclr

    precision = amb_clr / total_clr if total_clr > 0 else 0
    recall = amb_clr / total_amb if total_amb > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Clarification Decision Quality")
    print(f"  Total: {len(results)}")
    print(f"  Ambiguous: {total_amb} ({total_amb/len(results)*100:.1f}%)")
    print(f"  Clarified: {total_clr} ({total_clr/len(results)*100:.1f}%)")
    print(f"  Confusion Matrix:")
    print(f"    amb+clr={amb_clr}  amb+no_clr={amb_noclr}")
    print(f"    nonamb+clr={nonamb_clr}  nonamb+no_clr={nonamb_noclr}")
    print(f"  Precision: {precision:.3f} (of clarified, % truly ambiguous)")
    print(f"  Recall: {recall:.3f} (of ambiguous, % clarified)")
    print(f"  F1: {f1_score:.3f}")
    print(f"{'='*60}")

    # Classify clarify questions
    clr_questions = [r["clarify_question"] for r in results if r["has_clarify"]]
    if clr_questions:
        # Simple classification
        specific = sum(1 for q in clr_questions if any(w in q.lower() for w in ["which", "what kind", "what type", "whose", "what year", "what period"]))
        vague = sum(1 for q in clr_questions if any(w in q.lower() for w in ["more detail", "elaborate", "clarify", "specify"]))
        other = len(clr_questions) - specific - vague
        print(f"\n  Clarify Question Types:")
        print(f"    Specific (which/what kind/etc): {specific} ({specific/len(clr_questions)*100:.0f}%)")
        print(f"    Vague (more details/etc): {vague} ({vague/len(clr_questions)*100:.0f}%)")
        print(f"    Other: {other} ({other/len(clr_questions)*100:.0f}%)")

    # Save
    out_path = f"eval_results/clarify_precision_{ckpt.split('/')[-1]}.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "precision": precision, "recall": recall, "f1": f1_score,
                "amb_clr": amb_clr, "amb_noclr": amb_noclr,
                "nonamb_clr": nonamb_clr, "nonamb_noclr": nonamb_noclr,
                "total_clr": total_clr, "total_amb": total_amb,
            },
            "results": results
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
