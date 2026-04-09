"""Test DCR heuristic classification on real clarification examples."""

import re

def count_entity_density(text):
    """Estimate entity density: ratio of capitalized non-first-word tokens."""
    tokens = text.split()
    if len(tokens) <= 3:
        return 0.0
    entity_count = 0
    for t in tokens[1:]:
        if t[0].isupper() and len(t) > 2 and t.isalpha():
            entity_count += 1
        if t.startswith('"') or t.startswith("'") or t.startswith('('):
            entity_count += 1
    return entity_count / max(len(tokens) - 1, 1)

def classify_clarification(text):
    text = text.strip()
    if not text:
        return 'uncertain', 0.0
    text_lower = text.lower()
    tokens = text.split()
    n_tokens = len(tokens)
    entity_density = count_entity_density(text)

    if text_lower.startswith(('are you asking', 'do you mean', 'are you looking for', 'is the question', 'are you referring')):
        return 'intent', entity_density
    short_wh_starters = ('which', 'what', 'how much', 'how many', 'where', 'when', 'who', 'could', 'can')
    if n_tokens <= 15 and any(text_lower.startswith(w) for w in short_wh_starters):
        if entity_density < 0.3:
            return 'parameter', entity_density
        else:
            return 'search_like', entity_density
    number_count = len(re.findall(r'\d[\d,.]*', text))
    if number_count > 2 and n_tokens > 15:
        return 'search_like', entity_density
    if n_tokens > 40:
        return 'search_like', entity_density
    if entity_density > 0.25 and n_tokens > 12:
        return 'search_like', entity_density
    if n_tokens > 25:
        caps = len(re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', text))
        if caps >= 3:
            return 'search_like', entity_density
    if n_tokens <= 25:
        return 'parameter', entity_density
    return 'uncertain', entity_density

# PACIFIC clarifications from ar-best-200
pacific = [
    "Which year is being asked about?",
    "What is the end Balance of December 31, 2019?",
    "Is the question: What is the value of exercise per stock option as of August 29, 2019?",
    "assurance and related services that are reasonably related to the audit or review of the Company's financial statements",
    "What was the percentage change in amounts charged to cost and expenses of 4,569 to 7,392 in 2019?",
    "What was the percentage change in the Total provision for income taxes between 2018 and 2017?",
    "What is the future benefit payments for the U.S. plans from 2025-2029?",
    "Could you elaborate on what CRTC tangible benefits obligation you are referring to?",
    "What is the value of other current liabilities as a percentage of the total other liabilities for 2018?",
    "Are you asking: How many of Garo H. Armen's options would expire on October 16, 2027?",
    "What is the percentage change in 2018 from net cash provided by financing activities?",
    "What was the amount of time deposits in 2018 with original maturities of three months or less?",
    "What is the total net revenue in 2018 and 2019, including in-game net revenues, in millions?",
    "What was the net sales in the United States in 2017, in thousands?",
    "What is the percentage of Bell Wireless out of the Total BCE operating revenues in 2019?",
    "What is the difference in derivative instruments in current assets in that period from 2018 to 2019?",
    "Are you asking: What was the change in the percent of its revenue from 2017 to 2019?",
    "Are you asking: What is the percentage increase for the restaurant amount in 2018 over 2017?",
    "What is the percentage of constitution of costs among the total cost for 2018, including employee severance and related costs, strategic alternatives evaluation, Qdoba evaluation, and other costs?",
    "What was the average State and local income tax rate, net of federal tax benefits, in 2019?",
    "What is the ratio (in percentage) of total notes payable to total lease obligations as of May 26, 2019?",
    "What does the Number of Shares Acquired on Exercise column show in the Options Exercises table for 2018?",
]

# AbgCoQA cross-domain clarifications
abgcoqa = [
    "Who announced the recovery package of 180 million euros for the Ivory Coast?",
    "What happened after the House of Wangchuck reunited the country and established relations with the British Empire in the 19th century in Bhutan?",
    "What is the cause of the ongoing fighting in Abidjan?",
    "What did Mike find at the store that he wanted to buy",
    "Who reported the Pakistani Taliban's vow to carry out attacks against India to avenge the death of a man executed by Indian authorities for his role in the 2008 terrorist assault on Mumbai?",
]

print("=" * 80)
print("PACIFIC CLASSIFICATION (should be mostly intent/parameter)")
print("=" * 80)
counts = {}
for q in pacific:
    cls, ed = classify_clarification(q)
    counts[cls] = counts.get(cls, 0) + 1
    tag = "✅" if cls in ('intent', 'parameter') else ("⚠️" if cls == 'uncertain' else "❌")
    print(f"  {tag} [{cls:12s}] ed={ed:.2f}  {q[:65]}...")
print(f"\n  Distribution: {counts}")
full = counts.get('intent', 0) + counts.get('parameter', 0)
print(f"  Full bonus rate: {full}/{len(pacific)} = {full/len(pacific)*100:.0f}%")

print("\n" + "=" * 80)
print("AbgCoQA CLASSIFICATION (should be mostly search_like)")
print("=" * 80)
counts2 = {}
for q in abgcoqa:
    cls, ed = classify_clarification(q)
    counts2[cls] = counts2.get(cls, 0) + 1
    tag = "❌" if cls == 'search_like' else "⚠️"
    print(f"  {tag} [{cls:12s}] ed={ed:.2f}  {q[:65]}...")
print(f"\n  Distribution: {counts2}")
zero = counts2.get('search_like', 0)
print(f"  Zero bonus rate: {zero}/{len(abgcoqa)} = {zero/len(abgcoqa)*100:.0f}%")
