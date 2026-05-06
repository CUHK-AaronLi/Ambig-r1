"""Prompting baseline with multi-turn simulator interaction.
Same as RL eval but using SFT model with explicit clarify prompt.
Calls local Qwen simulator for clarification responses."""
import pandas as pd
import numpy as np
import re
import json
import sys
import requests
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

CLARIFY_SYSTEM = """You are a helpful agent answering questions about financial documents. Most questions can be answered directly from the table. Only ask a clarifying question when the question is genuinely ambiguous (e.g., missing year, unclear which category). Use these XML tags:
- <think>your reasoning</think>
- <clarify>your clarifying question</clarify>
- <answer>your final answer</answer>

Here are examples:

Example 1 — Clear question, answer directly:
User: What was the total revenue in 2019?
Assistant: <think>The question specifies 2019 and asks for total revenue. Clear.</think>
<answer>$1,496.5</answer>

Example 2 — Clear question, answer directly:
User: What is the net income for fiscal year 2018?
Assistant: <think>Year and metric are both specified.</think>
<answer>$342 million</answer>

Example 3 — Clear question despite seeming complex, answer directly:
User: What is the difference between 2019 and 2018 total assets?
Assistant: <think>Both years specified, calculation is straightforward.</think>
<answer>$1,205</answer>

Example 4 — Genuinely ambiguous (year missing), clarify:
User: What was the operating loss carryforward amount?
Assistant: <think>Multiple years in the table, question doesn't specify which.</think>
<clarify>Which year are you asking about?</clarify>

Example 5 — Genuinely ambiguous (category unclear), clarify:
User: What were the total costs?
Assistant: <think>Table has operating costs, COGS, and SGA separately.</think>
<clarify>Which type of costs — operating, COGS, or SGA?</clarify>

Now answer the following question. Answer directly unless genuinely ambiguous."""

SIM_URL = "http://10.10.211.117:8001/batch_generate"

def compute_token_f1(pred, ref):
    p_tok = pred.lower().split()
    r_tok = ref.lower().split()
    if not p_tok or not r_tok:
        return 0.0
    common = set(p_tok) & set(r_tok)
    if not common:
        return 0.0
    p = len(common) / len(p_tok)
    r = len(common) / len(r_tok)
    return 2 * p * r / (p + r)

def max_f1(pred, refs):
    if not refs:
        return 0.0
    return max(compute_token_f1(pred, r) for r in refs)

def extract_answer(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.I)
    if m:
        return m.group(1).strip()
    if '</user_response>' in text:
        last = text.rfind('</user_response>')
        after = text[last + len('</user_response>'):].strip()
        if after:
            return re.sub(r'^(?:answer|Answer)\s*[:：]\s*', '', after).strip()
    return text.strip()[-200:]

def call_simulator(question, clarify_q, context="", data_source="pacific"):
    """Call local Qwen simulator for clarification response."""
    try:
        queries = [{
            "question": question,
            "clarification_question": clarify_q,
            "unambiguous_question": "",
            "context": context,
            "data_source": data_source,
            "answer_hints": None,
        }]
        resp = requests.post(SIM_URL, json={"queries": queries, "return_scores": False}, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("result", [])
        if results:
            return results[0].get("response", "I need more information.")
        return "I need more information."
    except Exception as e:
        return f"Simulator error: {e}"

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "verl_checkpoints/sft-v4-ft/global_step_84"
    data_path = "scripts/data_process/data/mix_5ds/val_pacific.parquet"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    max_turns = 3
    # If "icl" in argv, override prompts with ICL few-shot system prompt
    use_icl = "icl" in sys.argv

    df = pd.read_parquet(data_path)
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} samples, max_turns={max_turns}, icl={use_icl}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    llm = LLM(model=ckpt, trust_remote_code=True, dtype="float16",
              max_model_len=4096, gpu_memory_utilization=0.80)
    # Stop at </clarify> or </answer> so we can call real simulator
    params_clr = SamplingParams(temperature=0.0, max_tokens=512,
                                stop=["</clarify>", "</answer>"])
    # For answer-only turns (after sim response), only stop at </answer>
    params_ans = SamplingParams(temperature=0.0, max_tokens=512,
                                stop=["</answer>"])

    results = []
    for i in range(len(df)):
        row = df.iloc[i]

        # Get ambiguity + gold
        rm = row.get("reward_model", {})
        if hasattr(rm, "tolist"): rm = rm.tolist()
        if isinstance(rm, str):
            try: rm = json.loads(rm)
            except: rm = {}
        gt = rm.get("ground_truth", {}) if isinstance(rm, dict) else {}
        if not isinstance(gt, dict): gt = {}
        is_amb = gt.get("req_clari", False)
        target = gt.get("target", [])
        if hasattr(target, "tolist"): target = target.tolist()
        if isinstance(target, str): target = [target]
        gold = [str(t).strip() for t in (target if isinstance(target, list) else [target]) if str(t).strip()]

        # Get original prompt
        p = row["prompt"]
        if hasattr(p, "tolist"): p = p.tolist()

        # Extract user question for simulator
        user_q = ""
        if isinstance(p, list):
            for msg in p:
                if msg.get("role") == "user":
                    user_q = msg.get("content", "")[-200:]

        # Build conversation with prompt
        if use_icl:
            # Replace system prompt with ICL few-shot, keep user content
            new_p = [{"role": "system", "content": CLARIFY_SYSTEM}]
            for msg in (p if isinstance(p, list) else []):
                if isinstance(msg, dict) and msg.get("role") != "system":
                    new_p.append(dict(msg))
            prompt_text = tokenizer.apply_chat_template(new_p, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)

        # Multi-turn loop
        full_response = ""
        n_clarify = 0
        for turn in range(max_turns + 1):
            cur_params = params_clr if turn == 0 else params_ans
            output = llm.generate([prompt_text], cur_params)
            response = output[0].outputs[0].text.strip()
            # Re-add closing tags that were consumed by stop sequences
            if '<clarify>' in response and '</clarify>' not in response:
                response += '</clarify>'
            if '<answer>' in response and '</answer>' not in response:
                response += '</answer>'
            full_response += response

            # With stop tokens, response ends at </clarify> or </answer>
            has_clarify_tag = '<clarify>' in response.lower()
            has_answer_tag = '<answer>' in response.lower()

            # Extract clarify question (stop token cuts off </clarify>)
            clr_match = re.search(r'<clarify>(.*?)$', response, re.DOTALL | re.I)
            if not clr_match:
                clr_match = re.search(r'<clarify>(.*?)</clarify>', response, re.DOTALL | re.I)

            if has_clarify_tag and not has_answer_tag and turn < max_turns:
                # Model wants to clarify — call simulator
                clr_q = clr_match.group(1).strip()
                sim_response = call_simulator(user_q, clr_q)
                n_clarify += 1

                # Append sim response and continue
                user_resp_text = f"\n\n<user_response>{sim_response}</user_response>\n\n"
                full_response += user_resp_text
                prompt_text = prompt_text + response + user_resp_text
            else:
                break

        # Extract answer and compute F1
        answer_text = extract_answer(full_response)
        f1 = max_f1(answer_text, gold) if gold else 0.0

        has_clarify = n_clarify > 0
        results.append({
            "idx": i, "is_ambiguous": bool(is_amb), "has_clarify": has_clarify,
            "n_clarify": n_clarify, "f1": f1, "answer_text": answer_text[:100],
        })

        if (i + 1) % 50 == 0:
            total = len(results)
            clr_n = sum(1 for r in results if r["has_clarify"])
            clr_f1 = np.mean([r["f1"] for r in results if r["has_clarify"]]) if clr_n > 0 else 0
            noclr_f1 = np.mean([r["f1"] for r in results if not r["has_clarify"]]) if (total - clr_n) > 0 else 0
            overall_f1 = np.mean([r["f1"] for r in results])
            print(f"[{i+1}/{len(df)}] F1={overall_f1:.3f} clr={clr_n}/{total} "
                  f"post_clr={clr_f1:.3f} no_clr={noclr_f1:.3f}")

    # Final
    total = len(results)
    clr_n = sum(1 for r in results if r["has_clarify"])
    clr_f1 = np.mean([r["f1"] for r in results if r["has_clarify"]]) if clr_n > 0 else 0
    noclr_f1 = np.mean([r["f1"] for r in results if not r["has_clarify"]]) if (total - clr_n) > 0 else 0
    overall_f1 = np.mean([r["f1"] for r in results])

    print(f"\n{'='*60}")
    print(f"Multi-Turn Prompting Baseline ({ckpt})")
    print(f"  Total: {total}, F1: {overall_f1:.3f}")
    print(f"  Clarified: {clr_n} ({clr_n/total*100:.1f}%)")
    print(f"  Post-clr F1: {clr_f1:.3f} (n={clr_n})")
    print(f"  No-clr F1: {noclr_f1:.3f} (n={total-clr_n})")

    # Precision
    amb_clr = sum(1 for r in results if r["is_ambiguous"] and r["has_clarify"])
    prec = amb_clr / clr_n if clr_n > 0 else 0
    print(f"  Precision: {prec:.3f}")
    print(f"{'='*60}")

    tag = "icl" if use_icl else "sft"
    out = f"eval_results/prompt_multiturn_{tag}_{n_samples}.json"
    with open(out, "w") as f:
        json.dump({"f1": overall_f1, "clr_rate": clr_n/total,
                    "post_clr_f1": clr_f1, "no_clr_f1": noclr_f1,
                    "precision": prec, "n": total, "results": results}, f, indent=2)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
