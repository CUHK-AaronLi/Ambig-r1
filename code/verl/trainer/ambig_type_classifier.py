#!/usr/bin/env python3
"""
AmbiguitType Conditioned Clarification (ATCC)

Phase 1: Create ambiguity-type annotated training data
- Classify each question by ambiguity type using a lightweight heuristic
- The heuristic uses question patterns + answer_type from extra_info

Phase 2: Implement ATCC in reward function
- CONDITIONAL clarify_bonus based on detected ambiguity type
- FACTUAL/INTERPRETIVE: reduced or zero clarify_bonus
- PARAMETER/RULES/SCOPE: full clarify_bonus

Phase 3: Train with ATCC on mixed 5ds data
- Start from ar-best-200 checkpoint
- Mix 5ds data
- ATCC reward function
"""

import re

# Ambiguity type classification heuristics
def classify_question_type(question: str, extra_info: dict = None) -> str:
    """
    Classify a question by ambiguity type.

    Returns: 'PARAMETER', 'RULES', 'SCOPE', 'FACTUAL', 'INTERPRETIVE', 'UNKNOWN'
    """
    if not question:
        return 'UNKNOWN'

    q = question.strip()
    q_lower = q.lower()
    tokens = q.split()
    n = len(tokens)

    # 1. Use answer_type from extra_info as a signal
    if extra_info and isinstance(extra_info, dict):
        answer_type = extra_info.get('answer_type', '')
        if answer_type == 'yesno':
            return 'RULES'  # SHARC: yes-no questions are rule-based
        elif answer_type == 'multi-span':
            # Could be PARAMETER or FACTUAL
            pass  # fall through to question analysis

    # 2. PARAMETER: asking for specific values/parameters
    param_patterns = [
        'which year', 'what year', 'how much', 'how many',
        'which company', 'which segment', 'what percentage',
        'which quarter', 'what is the value', 'in 201', 'in 202',
        'what is the total', 'what was the amount', 'how many of',
        'what is the balance', 'what was the revenue',
        'what is the change', 'what percentage of',
        'what was the percentage', 'what does the',
    ]
    if any(p in q_lower for p in param_patterns):
        return 'PARAMETER'

    # 3. RULES: legal/rule-following questions
    rule_patterns = [
        'chapter', 'can i', 'can someone', 'can the', 'can a',
        'do i', 'do you', 'does the', 'do the',
        'is the user', 'the user is', 'filing under',
        'applicable if', 'only if', 'provided that',
        'following conditions', 'eligible for', 'qualify for',
        'requirement', 'must i', 'must the',
    ]
    if any(p in q_lower for p in rule_patterns):
        return 'RULES'

    # 4. SCOPE: temporal/spatial scope
    scope_patterns = [
        'after ', 'before ', 'during ', 'between ',
        'from the period', 'in the period', 'over the period',
        'what happened after', 'what happened before',
    ]
    if any(p in q_lower for p in scope_patterns):
        return 'SCOPE'

    # 5. FACTUAL: asking for specific facts/sources/names
    factual_patterns = [
        'who announced', 'who said', 'who reported',
        'who is the', 'who was the', 'who were the',
        'when was the', 'when did the', 'when did x',
        'what year was', 'what happened when',
        'what did x say', 'who founded', 'who created',
        'who won', 'who played', 'who wrote',
    ]
    if any(p in q_lower for p in factual_patterns):
        return 'FACTUAL'

    # 6. INTERPRETIVE: subjective/judgment questions
    interpret_patterns = [
        'is this good', 'is it right', 'is it better',
        'is it ethical', 'is this ethical', 'is it appropriate',
        'should it be', 'is this the best', 'is it worth',
        'is really good', 'is it helpful', 'is it effective',
    ]
    if any(p in q_lower for p in interpret_patterns):
        return 'INTERPRETIVE'

    # 7. Short questions starting with who/what/where/when → likely FACTUAL
    if n <= 10 and q_lower.startswith(('who', 'what', 'where', 'when', 'how')):
        return 'FACTUAL'

    # 8. UNKNOWN: default
    return 'UNKNOWN'


def get_ambig_type_from_prompt(prompt_content: str) -> str:
    """Extract ambiguity type from full prompt content."""
    if not prompt_content:
        return 'UNKNOWN'

    # Try to extract the question from the prompt
    lines = prompt_content.split('\n')
    for line in reversed(lines):
        if 'question' in line.lower() or '?' in line:
            return classify_question_type(line)

    return classify_question_type(prompt_content[-200:])


def compute_atcc_reward(ambig_type: str, base_clarify_bonus: float = 0.15) -> float:
    """
    Compute clarification bonus based on ambiguity type.

    ATCC: Ambiguity-Type Conditioned Clarification

    - PARAMETER/RULES/SCOPE: full bonus (clarification helps)
    - FACTUAL/INTERPRETIVE: zero bonus (clarification hurts)
    - UNKNOWN: 50% bonus (conservative)
    """
    if ambig_type in ('PARAMETER', 'RULES', 'SCOPE'):
        return base_clarify_bonus  # Full reward
    elif ambig_type in ('FACTUAL', 'INTERPRETIVE'):
        return 0.0  # No reward for clarification on factual questions
    else:  # UNKNOWN
        return 0.5 * base_clarify_bonus  # Conservative


# Quick test
if __name__ == '__main__':
    test_cases = [
        ("What is the revenue in 2019?", "PARAMETER"),
        ("Can I file bankruptcy under Chapter 9?", "RULES"),
        ("What happened after Grasscannon?", "SCOPE"),
        ("Who announced the recovery package for Ivory Coast?", "FACTUAL"),
        ("Is this ethical?", "INTERPRETIVE"),  # subjective judgment
        ("How many helpers did he have?", "PARAMETER"),
        ("And after Grasscannon?", "SCOPE"),
        ("is really good in one class?", "INTERPRETIVE"),  # subjective quality
        ("Do I need a supplier document?", "RULES"),
    ]

    print("ATCC Classification Test:")
    print(f"{'Question':<55} {'Predicted':<15} {'Expected':<15} {'Bonus':<8}")
    print("-"*95)
    for q, expected in test_cases:
        predicted = classify_question_type(q)
        bonus = compute_atcc_reward(predicted)
        match = "✓" if predicted == expected else "?"
        print(f"{q[:53]:<55} {predicted:<15} {expected:<15} {bonus:.3f}  {match}")

    print("\nATCC Reward Table:")
    for t in ['PARAMETER', 'RULES', 'SCOPE', 'FACTUAL', 'INTERPRETIVE', 'UNKNOWN']:
        b = compute_atcc_reward(t)
        print(f"  {t:<15}: bonus={b:.3f}")
