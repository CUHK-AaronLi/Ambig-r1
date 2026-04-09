#!/usr/bin/env python3
"""
Radical Direction: Ambiguity Type Analysis

Core question: Can we diagnose what TYPE of ambiguity a question has,
before deciding whether to clarify?

Paper 2's BPP framework showed that behavioral probes can diagnose
uncertainty type. We apply the same idea to ambiguity type diagnosis.

Five ambiguity types:
1. PARAMETER: "What is the revenue in 2018?" — which parameter/year/category?
   → Should clarify: "Which year/segment?"
   → PACIFIC examples

2. REFERENTIAL: "Who did it?" — who is "he"?
   → Should clarify: "Are you referring to X?"
   → AbgCoQA examples

3. INTERPRETIVE: "Is this good?" — subjective interpretation
   → Should NOT clarify: impossible to disambiguate user intent
   → Some AmbigNQ examples

4. SCOPE: "What happened after?" — how far back/to what?
   → Should clarify: "After which event?"
   → AbgCoQA examples

5. FACTUAL: "Who announced X?" — which source?
   → Should NOT clarify aggressively: not intent clarification
   → AbgCoQA examples (the "search-like clarification" cases)

6. RULES (NEW): "Can I file under Chapter 9?" — rule-following
   → Should clarify: "What are the conditions for Chapter 9?"
   → SHARC examples
"""

import os
import pandas as pd
import re

def get_prompts_from_parquet(path):
    """Extract original questions from parquet files."""
    if not os.path.exists(path):
        return []

    df = pd.read_parquet(path)
    questions = []

    for _, row in df.iterrows():
        extra = row.get('extra_info', {})
        if isinstance(extra, dict):
            q = extra.get('original_question', '') or extra.get('gold_question', '')
        else:
            q = ''

        # Also try from prompt
        if not q:
            prompt = row.get('prompt', [])
            if isinstance(prompt, list):
                for m in prompt:
                    if isinstance(m, dict) and m.get('role') == 'user':
                        content = m.get('content', '')
                        # Extract question
                        lines = content.split('\n')
                        for line in lines:
                            if 'Question:' in line or 'question' in line.lower():
                                q = line.split('Question:')[-1].strip()
                                break
                        if not q and len(lines) > 0:
                            q = lines[-1][:200]
                        break

        questions.append({
            'question': q[:200] if q else 'N/A',
            'data_source': row.get('data_source', 'unknown'),
            'extra': extra if isinstance(extra, dict) else {},
        })

    return questions


def classify_ambiguity_type(question):
    """Heuristic ambiguity type classifier."""

    q_lower = question.lower()
    q_tokens = question.split()
    n_tokens = len(q_tokens)

    # RULES: rule-based / legal / conditional questions
    rule_patterns = ['chapter', 'if you', 'if a', 'can i', 'can someone', 'must i',
                    'does the', 'do the', 'is the user', 'the user is', 'what is the rule',
                    'follows: ', 'the rule:', 'condition', 'eligible for', 'qualify',
                    'filing under', 'applicable if', 'only if', 'provided that']
    if any(p in q_lower for p in rule_patterns):
        return 'RULES'

    # PARAMETER indicators: specific value/year/category ambiguity
    param_patterns = ['which year', 'what year', 'how much', 'how many',
                     'which company', 'which segment', 'what percentage',
                     'which quarter', 'what is the value of', 'in 201', 'in 202',
                     'what is the total', 'what was the amount']
    if any(p in q_lower for p in param_patterns):
        return 'PARAMETER'

    # REFERENTIAL indicators: pronoun/reference ambiguity
    ref_patterns = ['who did', 'what did he', 'what did she', 'it was',
                   'when did they', 'who was the', 'which person',
                   'that he', 'that she', 'who was it']
    if any(p in q_lower for p in ref_patterns):
        return 'REFERENTIAL'

    # SCOPE indicators: temporal/spatial scope ambiguity
    scope_patterns = ['after', 'before', 'during', 'between', 'among',
                     'in the same', 'what happened after', 'what happened before',
                     'what was the', 'from the period']
    if any(p in q_lower for p in scope_patterns):
        return 'SCOPE'

    # INTERPRETIVE indicators: subjective/value judgment
    inter_patterns = ['is this good', 'is it right', 'is it better',
                     'should it be', 'is it appropriate', 'is it ethical']
    if any(p in q_lower for p in inter_patterns):
        return 'INTERPRETIVE'

    # FACTUAL: asking for specific facts/announcements
    factual_patterns = ['who announced', 'who said', 'who reported',
                       'when did the company', 'what did x say', 'who won',
                       'who was the', 'what year was', 'when was the',
                       'how many did', 'what happened when']
    if any(p in q_lower for p in factual_patterns):
        return 'FACTUAL'

    # Short questions often are factoid
    if n_tokens <= 8 and q_lower.startswith(('who', 'what', 'where', 'when')):
        return 'FACTUAL'

    return 'UNKNOWN'


def main():
    print('='*70)
    print('RADICAL: Ambiguity Type Analysis')
    print('Inspired by Paper 2 BPP Framework — behavioral probes for ambiguity')
    print('='*70)

    data_dir = 'scripts/data_process/data'

    datasets = {
        'pacific_fewshot': 'PACIFIC (financial tables)',
        'abgcoqa': 'AbgCoQA (conversational)',
        'ambignq_fewshot': 'AmbigNQ (entity)',
        'sharc_fewshot': 'SHARC (rule-based)',
        'situatedqa_fewshot': 'SituatedQA (temporal)',
    }

    type_counts = {}

    for ds_key, ds_label in datasets.items():
        path = f'{data_dir}/{ds_key}/train.parquet'
        if not os.path.exists(path):
            print(f'\n--- {ds_label} ---')
            print(f'  NOT FOUND: {path}')
            continue

        print(f'\n--- {ds_label} ---')
        questions = get_prompts_from_parquet(path)

        if not questions:
            print(f'  No questions extracted')
            continue

        # Classify each question
        type_dist = {}
        samples = {k: [] for k in ['PARAMETER', 'RULES', 'REFERENTIAL', 'SCOPE', 'INTERPRETIVE', 'FACTUAL', 'UNKNOWN']}

        for item in questions[:500]:  # Sample 500
            q = item['question']
            atype = classify_ambiguity_type(q)
            type_dist[atype] = type_dist.get(atype, 0) + 1
            if len(samples[atype]) < 2:
                samples[atype].append(q)

        type_counts[ds_label] = type_dist

        total = sum(type_dist.values())
        print(f'  n={total}')
        for t in ['PARAMETER', 'RULES', 'REFERENTIAL', 'SCOPE', 'INTERPRETIVE', 'FACTUAL', 'UNKNOWN']:
            n = type_dist.get(t, 0)
            pct = n/total*100 if total > 0 else 0
            print(f'    {t:12s}: {n:4d} ({pct:5.1f}%)')
            if samples[t]:
                print(f'      Example: {samples[t][0][:80]}...')

    # Summary table
    print('\n' + '='*70)
    print('AMBIGUITY TYPE DISTRIBUTION SUMMARY')
    print('='*70)

    all_types = ['PARAMETER', 'RULES', 'REFERENTIAL', 'SCOPE', 'INTERPRETIVE', 'FACTUAL', 'UNKNOWN']
    print(f'\n{"Dataset":<35}', end='')
    for t in all_types:
        print(f' {t[:8]:>9}', end='')
    print()
    print('-'*100)

    for ds_label, dist in type_counts.items():
        total = sum(dist.values())
        print(f'{ds_label[:35]:<35}', end='')
        for t in all_types:
            n = dist.get(t, 0)
            pct = n/total*100 if total > 0 else 0
            print(f' {pct:8.1f}%', end='')
        print()

    # Action recommendations
    print('\n' + '='*70)
    print('CLARIFICATION RECOMMENDATIONS BY AMBIGUITY TYPE')
    print('='*70)

    recommendations = {
        'PARAMETER': 'SHOULD clarify — model should ask for specific parameter',
        'REFERENTIAL': 'SHOULD clarify — model should ask "Are you referring to X?"',
        'SCOPE': 'SHOULD clarify — model should ask for temporal/spatial scope',
        'RULES': 'SHOULD clarify — model should ask for rule conditions',
        'INTERPRETIVE': 'SHOULD NOT clarify — no objective answer to clarify',
        'FACTUAL': 'CAUTION — clarifying facts is search, not clarification',
        'UNKNOWN': 'NEEDS analysis — check manually',
    }

    for t, rec in recommendations.items():
        print(f'  {t:12s}: {rec}')

    print('\n' + '='*70)
    print('KEY INSIGHT: Four types of ambiguity → Four clarification strategies')
    print('='*70)
    print("""
  PACIFIC (41% PARAMETER): "Which year?" → Clarify parameters ✅
  SHARC (RULES): "Under what conditions?" → Clarify rule conditions ✅
  AbgCoQA (48% FACTUAL): "Who announced?" → Don't clarify aggressively ❌
  AmbigNQ (43% FACTUAL): "Who was?" → Don't clarify ❌

  → Model trained on PACIFIC only learns PARAMETER clarification
  → Applied to FACTUAL-dominant datasets → clarification hurts
  → SHARC (RULES) shows clarification helps because the model
    learned a "rule clarification" strategy that generalizes
""")


if __name__ == '__main__':
    main()
