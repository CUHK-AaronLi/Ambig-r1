"""Test DCR v2 heuristic with entity + content overlap."""

import re

COMMON_WORDS = {'The', 'What', 'Which', 'How', 'Are', 'Was', 'Were',
                'This', 'That', 'These', 'Those', 'And', 'For', 'From',
                'With', 'Not', 'Can', 'Could', 'Does', 'Did', 'Has', 'Have'}
STOP_WORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
             'to', 'for', 'of', 'and', 'or', 'but', 'it', 'this', 'that',
             'these', 'those', 'what', 'which', 'who', 'how', 'when', 'where',
             'why', 'do', 'does', 'did', 'has', 'have', 'had', 'be', 'been',
             'being', 'not', 'no', 'its', 'his', 'her', 'their', 'my', 'your'}

def classify_v2(clarify_text, original_question=''):
    text = clarify_text.strip()
    if not text:
        return 'uncertain', {}
    text_lower = text.lower()
    tokens = text.split()
    n_tokens = len(tokens)
    info = {'n_tokens': n_tokens}

    if text_lower.startswith(('are you asking', 'do you mean', 'are you looking for',
                               'is the question', 'are you referring')):
        return 'intent', info

    short_wh_starters = ('which', 'what', 'how much', 'how many', 'where', 'when', 'who', 'could', 'can')
    if n_tokens <= 15 and any(text_lower.startswith(w) for w in short_wh_starters):
        return 'parameter', info

    if original_question:
        orig_lower = original_question.lower()
        orig_entities = set(re.findall(r'[A-Z][a-z]{2,}', original_question))
        clarify_entities = set(re.findall(r'[A-Z][a-z]{2,}', clarify_text))
        new_entities = clarify_entities - orig_entities - COMMON_WORDS
        info['new_entities'] = new_entities

        if len(new_entities) >= 3:
            info['reason'] = f'new_entities={len(new_entities)}'
            return 'search_like', info
        if len(new_entities) >= 2 and n_tokens > 20:
            info['reason'] = f'new_entities={len(new_entities)}+long'
            return 'search_like', info

        # Content overlap
        clarify_content = set(w for w in text_lower.split() if w not in STOP_WORDS and len(w) > 2)
        orig_content = set(w for w in orig_lower.split() if w not in STOP_WORDS and len(w) > 2)
        if orig_content:
            overlap = clarify_content & orig_content
            overlap_ratio = len(overlap) / len(orig_content)
            info['overlap_ratio'] = overlap_ratio
            info['overlap_words'] = overlap
            orig_len = len(original_question.split())
            clar_len = n_tokens

            if overlap_ratio < 0.3 and clar_len > orig_len * 1.5:
                info['reason'] = f'low_overlap={overlap_ratio:.2f}+long={clar_len}>{orig_len}*1.5'
                return 'search_like', info
            if overlap_ratio < 0.5 and clar_len > orig_len * 2.0:
                info['reason'] = f'med_overlap={overlap_ratio:.2f}+very_long={clar_len}>{orig_len}*2.0'
                return 'search_like', info

    number_count = len(re.findall(r'\d[\d,.]*', text))
    if number_count > 2 and n_tokens > 20:
        return 'search_like', info
    if n_tokens > 40:
        return 'search_like', info
    if n_tokens > 25:
        caps = len(re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', text))
        if caps >= 3:
            return 'search_like', info
    if n_tokens <= 25:
        return 'parameter', info
    return 'uncertain', info


# Test cases with (clarification, original_question, expected)
test_cases = [
    # PACIFIC - should be intent/parameter
    ("What is the end Balance of December 31, 2019?",
     "What is the balance as of December 31, 2019?", "parameter"),
    ("What was the percentage change in amounts charged to cost and expenses of 4,569 to 7,392 in 2019?",
     "What is the change in cost and expenses?", "parameter"),
    ("Could you elaborate on what CRTC tangible benefits obligation you are referring to?",
     "What are the tangible benefits?", "parameter"),
    ("Are you asking: How many of Garo H. Armen's options would expire on October 16, 2027?",
     "How many options would expire?", "intent"),
    ("What is the total net revenue in 2018 and 2019, including in-game net revenues, in millions?",
     "What is the net revenue?", "parameter"),
    ("What is the percentage of Bell Wireless out of the Total BCE operating revenues in 2019?",
     "What is the percentage of BCE wireless revenue?", "parameter"),

    # AbgCoQA - should be search_like
    ("Who announced the recovery package of 180 million euros for the Ivory Coast?",
     "What was announced for the Ivory Coast?", "search_like"),
    ("What happened after the House of Wangchuck reunited the country and established relations with the British Empire in the 19th century in Bhutan?",
     "What happened after that?", "search_like"),
    ("What is the cause of the ongoing fighting in Abidjan?",
     "What is the cause of the fighting?", "search_like"),
    ("What did Mike find at the store that he wanted to buy",
     "What did Mike find?", "search_like"),
    ("Who reported the Pakistani Taliban's vow to carry out attacks against India to avenge the death of a man executed by Indian authorities for his role in the 2008 terrorist assault on Mumbai?",
     "Who reported the vow?", "search_like"),

    # AmbigNQ
    ("What year did the film starring Tom Hanks and directed by Steven Spielberg come out?",
     "What year did the film come out?", "parameter"),  # or search_like
]

print("=" * 90)
print("DCR v2 Classification Test")
print("=" * 90)

correct = 0
total = len(test_cases)
for clarify, orig, expected in test_cases:
    cls, info = classify_v2(clarify, orig)
    is_correct = (cls == expected or (expected == 'search_like' and cls == 'search_like') or
                  (expected in ('intent', 'parameter') and cls in ('intent', 'parameter')))
    correct += is_correct
    tag = "✅" if is_correct else "❌"

    overlap = info.get('overlap_ratio', None)
    new_e = info.get('new_entities', set())
    reason = info.get('reason', '-')

    ne_str = f"new=[{', '.join(list(new_e)[:4])}]" if new_e else ""
    ov_str = f"overlap={overlap:.2f}" if overlap is not None else ""
    print(f"  {tag} [{cls:12s}] exp={expected:12s} {ne_str:35s} {ov_str:15s} {clarify[:50]}...")

print(f"\n  Accuracy: {correct}/{total} = {correct/total*100:.0f}%")
if correct < total:
    print("  (Note: Some mismatches are acceptable if the classification is reasonable)")

# Key comparison
print("\n" + "=" * 90)
print("v1 (no context) vs v2 (with original question)")
print("=" * 90)
for clarify, orig, expected in test_cases:
    cls_v1, _ = classify_v2(clarify, '')
    cls_v2, info = classify_v2(clarify, orig)
    improved = cls_v1 != 'search_like' and cls_v2 == 'search_like'
    correct_v2 = cls_v2 == expected or (expected in ('intent', 'parameter') and cls_v2 in ('intent', 'parameter'))
    arrow = "⬆️" if improved else ("✓" if correct_v2 else "?")
    print(f"  {arrow} v1={cls_v1:12s} → v2={cls_v2:12s} (exp={expected:12s}) {clarify[:50]}...")
