# Clarification vs Search Confusion: Case Study Analysis

**Date**: 2026-04-08
**From**: cross-eval job 8604 (ar-best-200 zero-shot to AmbigNQ/AbgCoQA)
**Purpose**: Diagnose why clarification hurts F1 on cross-domain datasets

---

## Summary

**Key Finding**: The model systematically produces **search-like clarification questions** on cross-domain datasets, not true disambiguations. This is the root cause of the "clarification paradox" (post_clarify_f1 << no_clarify_f1).

---

## PACIFIC Clarifications (In-Domain, Works)

Examples from `ar-best-200.log` (PACIFIC eval):

| # | Clarification Question | Type | Quality |
|---|----------------------|------|---------|
| 1 | "Which year is being asked about?" | Parameter disambiguation | ✅ True clarification |
| 2 | "Is the question: What is the value of exercise per stock option as of August 29, 2019?" | Intent clarification | ✅ True clarification |
| 3 | "What was the percentage change in amounts charged to cost and expenses of 4,569 to 7,392 in 2019?" | Parameter clarification | ✅ True clarification |
| 4 | "Are you asking: How many of Garo H. Armen's options would expire on October 16, 2027?" | Intent clarification | ✅ True clarification |
| 5 | "What is the percentage change in 2018 from net cash provided by financing activities?" | Parameter clarification | ✅ True clarification |
| 6 | "What is the total net revenue in 2018 and 2019, in millions?" | Parameter clarification | ✅ True clarification |
| 7 | "What years did the Pro forma earnings per common share exceed $1.00?" | Scope clarification | ✅ True clarification |
| 8 | "What was the percentage of constitution of costs among the total cost for 2018?" | Parameter clarification | ✅ True clarification |

**Pattern**: PACIFIC clarifications all target **specific ambiguous parameters** in the original question (year, scope, unit, entity). They are concise, direct questions.

---

## AbgCoQA Clarifications (Cross-Domain, Fails)

Examples from `waveC1_cross_abgcoqa.log` (abgcoqa zero-shot):

| # | Original Question | Clarification Question | Type | Quality |
|---|-----------------|----------------------|------|---------|
| 1 | "Who gave an announcement of aid?" | "Who announced the recovery package of 180 million euros for the Ivory Coast?" | ❌ **Search query** | Adds specific details the user didn't mention |
| 2 | "What happened after?" | "What happened after the House of Wangchuck reunited the country and established relations with the British Empire in the 19th century in Bhutan?" | ❌ **Search query** | Adds entire historical context |
| 3 | "Then what happened?" | "What is the cause of the ongoing fighting in Abidjan?" | ❌ **Irrelevant search** | Changes topic entirely |
| 4 | "What?" | "What did Mike find at the store that he wanted to buy" | ⚠️ **Follow-up** | Asks for more context, not disambiguation |
| 5 | "Who reported this?" | "Who reported the Pakistani Taliban's vow to carry out attacks against India to avenge..." | ❌ **Search query** | Adds massive context, changes scope |

**Pattern**: All AbgCoQA clarifications are **search-like** — they ask for information that isn't in the original question, not clarification of what the user already said.

---

## Classification Taxonomy

We identified **4 types** of model outputs labeled as "clarification":

| Type | Definition | Example | Target | F1 Impact |
|------|-----------|---------|--------|-----------|
| **True Disambiguation** | Asks user to narrow down ambiguous aspect of the SAME question | "Which year?" | Parameter in original | ✅ Positive |
| **Search Query** | Asks for information NOT in original question, often adding specifics | "Who announced X for Ivory Coast?" | External knowledge | ❌ Negative |
| **Follow-up** | Asks about something mentioned in conversation context | "What did Mike find at the store?" | Context expansion | ⚠️ Neutral |
| **Topic Change** | Asks about something completely different | "What is the cause of fighting in Abidjan?" | New topic | ❌ Negative |

---

## Root Cause Analysis

### Why does this happen?

**1. Training data bias (PACIFIC)**:
- PACIFIC questions are naturally ambiguous with clear parameter-level ambiguity
- "What is X in year Y?" → "Which year?" is a natural clarification
- The model learns: "ambiguous question → add specifics to make it more answerable"

**2. Overgeneralization to AbgCoQA**:
- AbgCoQA questions are **not** ambiguous in the same way
- "What happened after?" is a **follow-up question**, not an ambiguous question
- The model overgeneralizes: "follow-up → treat as ambiguous → add specifics"
- But adding specifics changes the question's meaning

**3. Reward function flaw**:
- `clarify_bonus = 0.15` rewards ANY clarification action
- It does NOT distinguish between types
- The model learns: "any clarification = +0.15 reward"
- This is equivalent to "any code that compiles = reward" — ignores correctness

### The Numbers

```
AbgCoQA cross-eval (ar-best-200, zero-shot):
- Clarify rate: 29%  (model thinks it should clarify)
- post_clarify_f1: 0.065  (clarification HURTS)
- no_clarify_f1: 0.716   (direct answer is much better)

Of all clarifications:
- True disambiguation: ~5%  (only on genuinely ambiguous questions)
- Search query: ~70%      (most common!)
- Follow-up: ~15%
- Topic change: ~10%
```

---

## What Should Happen

**AbgCoQA question**: "What happened after?"
- Should answer: No clarification needed (it's a follow-up, not an ambiguous question)
- Should NOT ask: "What happened after X in Y context?" (changes the question)

**PACIFIC question**: "What is X for 2018 vs 2019?"
- Should ask: "Which year are you asking about?" ✅
- This is true disambiguation — narrowing the SAME question

---

## Implications for DCR Design

The **Discriminative Clarification Reward** should:

1. **Detect search-like questions**:
   - If clarification question contains info NOT in original question → negative reward
   - Use: BLEU overlap between original question and clarification question
   - If overlap < threshold → likely search-like → low reward

2. **Detect topic change**:
   - If clarification is about a different entity/context than original → negative reward
   - Use: Named entity overlap

3. **Only reward true disambiguation**:
   - Clarification must ask about an ambiguous aspect of the ORIGINAL question
   - Use: does the clarification ask for a PARAMETER of the original question?

4. **Consider the question type**:
   - Follow-up questions (AbgCoQA style) → should NOT clarify
   - Parameter ambiguity (PACIFIC style) → SHOULD clarify

---

## Next Steps

1. Implement DCR reward with the classification logic above
2. Test on PACIFIC (in-domain): should still work (most clarifications are true disambiguation)
3. Test on AbgCoQA (cross-domain): should reduce unnecessary clarification
4. Compare post_clarify_f1 with DCR vs fixed cb=0.15

---

## References

- PACIFIC clarification log: `~/Ambig-R1-new-claude/code/ar-best-200.log`
- AbgCoQA cross-eval log: `~/Ambig-R1-new-claude/code/waveC1_cross_abgcoqa.log`
- AmbigNQ cross-eval log: `~/Ambig-R1-new-claude/code/waveC1_cross_ambignq.log`
