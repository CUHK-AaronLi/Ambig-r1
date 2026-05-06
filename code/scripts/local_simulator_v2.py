#!/usr/bin/env python3
"""
Local Clarification Simulator v2 — fixed answer leakage.

Changes from v1:
- REMOVED answer_hints from prompt (was directly leaking gold answer)
- REMOVED context/table from prompt (simulator could compute answer from table)
- Added few-shot examples showing correct behavior
- Simulator only knows: original question + what user actually meant + clarify question
"""
import os, json, logging, argparse
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VLLM_CLIENT = None
VLLM_MODEL = None

class ClarifyQuery(BaseModel):
    question: str
    clarification_question: str
    unambiguous_question: Optional[str] = None
    context: str = ""
    data_source: str = "generic"
    reference_question: Optional[str] = None
    reference_answer: Optional[str] = None
    answer_hints: Optional[List[str]] = None
    missing_details: Optional[List[dict]] = None  # IntentionGym persona

class BatchQueryRequest(BaseModel):
    queries: List[ClarifyQuery]
    return_scores: bool = False

SYSTEM_PROMPT = """You simulate a user who asked a question to a financial QA assistant.

## ABSOLUTE RULES (violating any = failure):
1. You do NOT know the answer. NEVER include numbers, dollar amounts, percentages, or computed values.
2. Answer ONLY the ONE dimension asked. If asked "which year?" → say ONLY the year. If asked "which metric?" → say ONLY the metric. NEVER combine multiple pieces of info.
3. If the assistant's question is vague or too broad (e.g., "more details?", "can you elaborate?", "tell me more"), push back: ask THEM to be specific. Do NOT dump your info.
4. If the question is irrelevant to your original question, say so.
5. Keep response under 20 words.

## GOOD responses (follow these exactly):

Q: "What is the revenue?" | Meant: "Revenue in fiscal year 2019?"
Assistant: "Which year?" → "Fiscal year 2019."
Assistant: "What metric?" → "Revenue."
Assistant: "Can you provide more details?" → "What specifically do you need to know?"
Assistant: "What is the revenue amount?" → "I don't know, that's what I'm asking."

Q: "What is the increase in gross profit in 2019?" | Meant: "Increase from Q1 to Q2?"
Assistant: "Which period?" → "From Q1 to Q2."
Assistant: "Which metric?" → "Gross profit."
Assistant: "Can you tell me more?" → "More about what? Please ask a specific question."

Q: "What is the change in expenses?" | Meant: "Change in operating expenses from 2018 to 2019?"
Assistant: "What kind of expenses?" → "Operating expenses."
Assistant: "Which years?" → "From 2018 to 2019."
Assistant: "Give me all the details" → "What do you need to know specifically?"

## BAD responses (NEVER do these):

BAD: "From Q1 to Q2 in 2019, the gross profit increased." ← combined period + metric + direction
BAD: "Operating expenses, specifically from 2018 to 2019." ← combined metric + period
BAD: "The increase was $13,407." ← included a number
BAD: "I want to know about long-term debt for 2019." ← gave both metric and year when only asked one
BAD: "Sure, here are all the details: operating expenses from 2018 to 2019." ← dumped everything

## For vague/broad questions from assistant:
"Can you provide more details?" → "What specifically do you want to know?"
"Tell me more" → "What aspect are you asking about?"
"Can you elaborate?" → "Could you ask a more specific question?"
"Give me all the information" → "I'd prefer you ask specific questions."

## For irrelevant questions:
"What is your favorite color?" → "That's not relevant to my question."
"Do you like pizza?" → "That's not related. Can you help with my question?"
"""

def create_simulation_prompt(query: ClarifyQuery) -> str:
    """Build user prompt — NO context, NO answer hints."""
    parts = []

    # Only tell simulator what the user MEANT (their intent)
    if query.unambiguous_question and query.unambiguous_question != query.question:
        parts.append(f'Your original question: "{query.question}"')
        parts.append(f'What you actually meant: "{query.unambiguous_question}"')
    else:
        parts.append(f'Your question: "{query.question}"')

    parts.append(f'The assistant asks: "{query.clarification_question}"')
    parts.append("How do you respond? (Brief, natural, NO numbers or data values)")

    return "\n".join(parts)


INTENTIONGYM_PROMPT = """You are a human user who gave a vague task to an AI assistant. The assistant is asking a clarification question. Answer briefly and naturally using ONLY the preferences below when relevant.

Your task: {task}

Your preferences (only answer what is asked, do NOT volunteer extra info):
{persona_str}

Rules:
- Answer ONLY the specific question asked
- Be brief (1-2 sentences)
- If the question is not about any preference, make up a reasonable answer
- Never reveal the full list of preferences"""


def _simulate_intentiongym(query: ClarifyQuery) -> str:
    """IntentionGym persona-based simulator."""
    details = query.missing_details
    if hasattr(details, "tolist"):
        details = details.tolist()

    # Build persona from missing_details options (pick first option as "chosen")
    persona_lines = []
    for d in details:
        desc = d.get("description", "")
        opts = d.get("options", [])
        if hasattr(opts, "tolist"):
            opts = opts.tolist()
        chosen = opts[0] if opts else "no preference"
        persona_lines.append(f"- {desc}: {chosen}")

    persona_str = "\n".join(persona_lines)
    task = query.question.strip()
    clr_q = query.clarification_question.strip()

    system_msg = INTENTIONGYM_PROMPT.format(task=task, persona_str=persona_str)
    try:
        completion = VLLM_CLIENT.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"The assistant asks: {clr_q}\nAnswer briefly:"}
            ],
            max_tokens=100,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"IntentionGym sim error: {e}")
        return "I am not sure, please pick a reasonable default."

def simulate_user_response(query: ClarifyQuery) -> str:
    try:
        prompt = create_simulation_prompt(query)
        completion = VLLM_CLIENT.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3,
        )
        response = completion.choices[0].message.content.strip()

        # Post-processing: strip any leaked numbers if answer_hints provided
        if query.answer_hints:
            for hint in query.answer_hints:
                hint_str = str(hint).strip()
                if hint_str and len(hint_str) > 1 and hint_str in response:
                    logger.warning(f"Answer leakage detected: '{hint_str}' in response '{response}'. Replacing.")
                    response = response.replace(hint_str, "[REDACTED]")

        return response
    except Exception as e:
        logger.error(f"vLLM error: {e}")
        return "I'm not sure, could you rephrase?"

@app.post("/batch_generate")
def generate_batch_response(request: BatchQueryRequest):
    results = []
    for query in request.queries:
        response = simulate_user_response(query)
        results.append({
            "response": response,
            "question": query.question,
            "clarification_question": query.clarification_question,
            "unambiguous_question": query.unambiguous_question,
            "context": "",
            "data_source": query.data_source,
        })
    return {"result": results, "return_scores": request.return_scores}

@app.get("/health")
def health():
    return {"status": "ok", "model": VLLM_MODEL}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8002/v1")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    VLLM_CLIENT = OpenAI(base_url=args.vllm_url, api_key="dummy")
    VLLM_MODEL = args.model_name
    logger.info(f"Local simulator v2: vLLM at {args.vllm_url}, model={args.model_name}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
