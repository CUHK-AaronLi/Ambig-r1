# Radical Direction: Ambiguity Type Analysis — Results

## Key Finding: 4 Ambiguity Types, Each Needs Different Strategy

From heuristic analysis of 500 examples per dataset:

| Dataset | PARAMETER | RULES | FACTUAL | CLARIFY_HELP? |
|--------|-----------|-------|---------|----------------|
| PACIFIC | 42.4% | 2.8% | 7.6% | ✅ Yes |
| SHARC | 0% | 50%+ | 0% | ✅ Yes (+0.116 F1) |
| AbgCoQA | 2.6% | 2.2% | **47.6%** | ❌ No (-0.651 F1) |
| AmbigNQ | 4.2% | 4.0% | **47.0%** | ❌ No (-0.118 F1) |
| SituatedQA | 4.6% | 0.8% | **51.4%** | ❌ No (-0.154 F1) |

## Why SHARC Works Despite Being "RULES"

SHARC is 50% RULES-type questions ("Can I file bankruptcy under Chapter 9?").
The model trained on PACIFIC (PARAMETER) somehow generalizes to SHARC (RULES).
This suggests:
- The IG reward teaches the model a "general clarification" pattern
- This pattern happens to work on RULES-type because RULES questions ask about conditions
- The clarification strategy generalizes from PARAMETER to RULES

But it FAILS on FACTUAL:
- FACTUAL questions ("Who announced?") when clarified produce SEARCH-like responses
- The GPT-4o simulator answers FACTUAL clarification with facts instead of intent-clarification
- This corrupts the answer trajectory

## Proposed Radical Solution: Ambiguity-Type Aware Agent

Inspired by Paper 2's BPP framework — behavioral probes for uncertainty diagnosis.

```
Before deciding whether to clarify:
1. Run ambiguity-type classifier (lightweight heuristic or small model)
2. If PARAMETER/RULES/SCOPE → clarify aggressively (current behavior)
3. If FACTUAL → DON'T clarify OR clarify very carefully (e.g., "Do you mean X or Y?")
4. If UNKNOWN → use default (don't clarify)
```

### Implementation Options

**Option 1: Pre-classification filter (simplest)**
- Before rollout, classify each question's ambiguity type
- Set clarify_prob=0.9 for PARAMETER/RULES, clarify_prob=0.1 for FACTUAL

**Option 2: Ambiguity-type conditioned reward**
- Different IG reward for different ambiguity types
- FACTUAL questions get lower clarify_bonus

**Option 3: Behavioral probe (inspired by BPP)**
- Generate 3 paraphrases of the question
- If answers are consistent → PARAMETER/RULES → clarify
- If answers are inconsistent → FACTUAL → don't clarify aggressively
- This is the BPP analog for ambiguity type

## Connection to Paper 2 (BPP)

Paper 2: BPP uses paraphrase consistency → diagnose LLM's own uncertainty type
Paper 1 v2: Paraphrase consistency → diagnose question's ambiguity type

Both use the same behavioral probe technique, but:
- BPP: probes the LLM's knowledge
- Our version: probes the question's structure

This creates a clean synergy between the two papers.

## Action Plan

Phase 1 (Today): Write ambiguity-type classifier
Phase 2 (Tomorrow): Run oracle experiment — if we knew ambiguity type, how much would F1 improve?
Phase 3 (If oracle positive): Implement ambiguity-type conditioned agent
