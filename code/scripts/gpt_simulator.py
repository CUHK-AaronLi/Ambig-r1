#!/usr/bin/env python3
"""
GPT Clarification Simulator
使用 Azure OpenAI 的 GPT-4 来模拟用户回答 clarification 问题
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
import logging
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Azure OpenAI配置
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "YOUR ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "YOUR API KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")

# 检查配置是否设置
if AZURE_ENDPOINT == "YOUR ENDPOINT" or AZURE_API_KEY == "YOUR API KEY":
    logger.error("=" * 80)
    logger.error("❌ Azure OpenAI configuration not set!")
    logger.error("Please set the following environment variables:")
    logger.error("  export AZURE_ENDPOINT='https://your-resource.openai.azure.com'")
    logger.error("  export AZURE_API_KEY='your-api-key'")
    logger.error("=" * 80)
    # 不退出，让服务启动，但会在实际调用时失败

# 初始化Azure OpenAI客户端
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    max_retries=0,  # SDK不重试，由我们的代码统一处理
)

# Pydantic models for request/response
class ClarifyQuery(BaseModel):
    question: str  # 模糊版问题
    clarification_question: str  # 代理提出的澄清问题
    unambiguous_question: Optional[str] = None  # 明确版问题
    context: str = ""
    data_source: str = "generic"
    reference_question: Optional[str] = None
    reference_answer: Optional[str] = None
    answer_hints: Optional[List[str]] = None


class SingleQueryRequest(ClarifyQuery):
    """Single clarify query request."""


class BatchQueryRequest(BaseModel):
    queries: List[ClarifyQuery]
    return_scores: bool = False


class SingleQueryResponse(BaseModel):
    response: str
    question: str
    clarification_question: str
    unambiguous_question: Optional[str] = None
    context: str
    data_source: str
    reference_question: Optional[str] = None
    reference_answer: Optional[str] = None


class BatchQueryResponse(BaseModel):
    result: List[Dict[str, Any]]
    return_scores: bool

def create_simulation_prompt(query: ClarifyQuery) -> str:
    """
    Create prompt for simulating user responses.

    v3: Dataset-aware branching for better simulator quality.
    Changes from v2:
    - PACIFIC-specific prompt uses table/document context (no unambiguous_question available)
    - Generic non-ambiguous prompt unchanged
    - Ambiguous prompt unchanged
    - answer_hints are NEVER included in any prompt (leakage prevention)
    """
    ambig_q = query.question.strip()
    unambig_q = (query.unambiguous_question or query.reference_question or "").strip()
    clarif_q = query.clarification_question.strip()
    context_block = query.context.strip()
    data_source = (query.data_source or "generic").strip().lower()

    # Check if the question is actually ambiguous
    is_same_question = (ambig_q.lower().rstrip('?').strip() == unambig_q.lower().rstrip('?').strip())

    if data_source == "pacific" and (is_same_question or not unambig_q):
        # PACIFIC: financial table QA with conversation history.
        # No unambiguous_question available — use document context to guide response.
        context_section = f"\nDocument you are referencing:\n{context_block}" if context_block else ""
        prompt = f"""You are a user who asked a question about a financial document:
"{ambig_q}"

An AI assistant is trying to help and asked:
"{clarif_q}"
{context_section}

Respond naturally as a real user would. Keep your response under 50 words.

Guidelines:
- You may specify WHICH part of the data you mean (e.g., which row, time period, or metric)
- You do NOT know the answer — never provide numerical results or computed values
- If the assistant asks for the answer itself, say "I don't know, that's why I'm asking"
- Only answer what is directly asked — do not volunteer extra information
- Do NOT use meta-language ("table", "document", "context") — speak conversationally

Examples:
- "I'm asking about the 2019 figures, not 2020."
- "Yes, I mean the net revenue, not gross."
- "I don't know the exact number — that's what I'm asking.\""""
        # Context already embedded, skip appending later
        return prompt

    elif is_same_question or not unambig_q:
        # Non-ambiguous case: the user's question is already clear
        prompt = f"""You are a user who asked the following question:
"{ambig_q}"

An AI assistant is trying to clarify your question and asked:
"{clarif_q}"

Your question is already clear and does not need clarification. Respond naturally as a user would.

Rules:
- Politely indicate that your question is straightforward and doesn't need clarification
- You may restate what you're asking in slightly different words
- Keep your response under 50 words
- Do NOT use any special format — just respond naturally

Example responses:
- "My question is straightforward — I'm just asking about [topic]."
- "I think my question is clear. I want to know [what you asked]."
- "No need to clarify — just answer the question as stated.\""""
    else:
        # Ambiguous case: provide disambiguating information
        # NOTE: answer_hints are intentionally NOT included in the prompt.
        # A real user knows what they meant (unambiguous_question) but does NOT
        # know the answer — that's why they're asking. answer_hints are only
        # used in clean_simulator_response() for leakage detection.

        prompt = f"""You are a user who asked the question: "{ambig_q}"

What you actually meant was: "{unambig_q}"

CRITICAL CONSTRAINTS:
1. You do NOT know the answer to your own question — you are just a user seeking help.
2. Keep your intended meaning private — do NOT directly repeat or reveal the unambiguous version of your question.
3. If the assistant's clarifying question is asking for the answer itself (e.g., "What year specifically?", "Which person?"), respond that this is beyond what you know — you are asking because you DON'T know the answer.

An AI assistant is trying to help you and asked this clarifying question:
"{clarif_q}"

Respond naturally as a real user would. Keep your response under 50 words. ONLY answer what is directly asked — do not volunteer extra information.

IMPORTANT — match your response to the quality of the clarification question:

1. **Precise question that identifies the exact ambiguity** (e.g., "Are you asking about X or Y?"):
   → Confirm briefly. "Yes, the Tracey Ullman shorts." is a perfect response. Do NOT add extra info.

2. **Somewhat relevant but vague question** (e.g., "Can you be more specific?"):
   → Give only a minimal hint. Do NOT dump all disambiguating details at once.

3. **Off-topic or nonsensical question**:
   → Say "That's not what I meant" with at most a small nudge toward the right direction.

4. **Question that asks for the answer itself** (e.g., "What year?" "Which person?"):
   → Refuse: "I don't know, that's why I'm asking." You may add a brief clarifying detail about your intent.

Key rules:
- It IS okay to respond with just "Yes, [brief confirmation]." if the question deserves it
- NEVER give away the answer to your question
- NEVER volunteer information that wasn't asked for
- NEVER use meta-language like "reference example", "unambiguous question", etc.

Examples:
- Q: "The Tracey Ullman shorts or the standalone series?" → "The Tracey Ullman shorts."
- Q: "Can you be more specific?" → "I mean the very first time they appeared on TV."
- Q: "What year did it air?" → "I don't know the year — that's what I'm asking."
- Q: "Are you asking about cooking?" → "No, that's not what I meant at all.\""""

    # Add context if available
    if context_block:
        prompt += f"\n\nAdditional context (for your reference, do not reveal): {context_block}"

    return prompt


def clean_simulator_response(response: str, gold_answers: Optional[List[str]] = None) -> str:
    """
    Post-process GPT-4o response to remove format artifacts and fix quality issues.

    Handles:
    - "ACTION : **ANSWER_CLARIFICATION** :" prefix from old prompt format
    - "reference example" language leaking from prompt
    - Answer leakage detection (Stengel-Eskin et al. 2025)
    - Meta-language leaks ("unambiguous question", etc.)

    Args:
        response: Raw GPT-4o response
        gold_answers: Optional list of gold answers for leakage detection
    """
    import re as _re

    # Strip ACTION format prefix (from old prompt or GPT-4o habit)
    response = _re.sub(
        r"^ACTION\s*:\s*\*\*ANSWER_CLARIFICATION\*\*\s*:\s*",
        "", response, flags=_re.IGNORECASE
    ).strip()

    # Strip surrounding quotes if GPT wrapped the response
    if len(response) >= 2 and response[0] == '"' and response[-1] == '"':
        response = response[1:-1].strip()

    # Replace "reference example" leak with natural response
    if _re.search(r"(irrelevant|not relevant)\s+(to\s+)?the\s+reference\s+example", response, _re.IGNORECASE):
        response = "That's not what I meant. Could you ask something more specific about my question?"

    # Replace other meta-language leaks
    if "unambiguous question" in response.lower() or "unambiguous version" in response.lower():
        response = "I'm not sure what you're asking. Could you rephrase your question?"

    # Answer leakage detection: if the response contains the gold answer,
    # replace with a generic redirect (prevents shortcut learning)
    if gold_answers:
        resp_lower = response.lower()
        for ans in gold_answers:
            ans_str = str(ans).strip().lower()
            if ans_str and len(ans_str) > 2 and ans_str in resp_lower:
                logger.warning(f"Answer leakage detected: '{ans_str}' found in response '{response[:80]}...'")
                response = "I don't know the exact answer — that's why I'm asking. But I can clarify what I mean if you ask more specifically."
                break

    # Truncate overly long responses (enforce ~50 word limit)
    words = response.split()
    if len(words) > 60:
        response = " ".join(words[:50]) + "..."

    # Ensure non-empty
    if not response.strip():
        response = "I'm not sure how to answer that. Could you ask a more specific question?"

    return response.strip()

def simulate_user_response(query: ClarifyQuery, max_retries: int = 1) -> str:
    """
    使用Azure OpenAI模拟用户回答clarification问题，带重试机制
    """
    # 输入验证
    if not query.question or not query.clarification_question:
        logger.warning("Missing required parameters: question or clarification_question")
        return "[System] Unable to process clarification."

    prompt = create_simulation_prompt(query)

    # 只重试一次，快速失败
    for attempt in range(max_retries + 1):  # 0次重试 = 尝试1次，1次重试 = 尝试2次
        try:
            completion = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": "You are simulating a user who asked a question. "
                                   "Respond naturally and concisely (under 50 words). "
                                   "Only answer what is asked — do not volunteer extra information. "
                                   "You do NOT know the answer to your question — never provide it."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150,
                temperature=0.3,
            )

            response = completion.choices[0].message.content.strip()
            # Pass answer_hints for leakage detection
            gold_answers = None
            if hasattr(query, 'answer_hints') and query.answer_hints:
                gold_answers = query.answer_hints
            response = clean_simulator_response(response, gold_answers=gold_answers)
            return response

        except (APIConnectionError, ConnectionError) as e:
            # 详细记录连接错误信息，帮助诊断问题
            error_details = {
                "type": type(e).__name__,
                "message": str(e),
                "endpoint": AZURE_ENDPOINT,
                "deployment": AZURE_DEPLOYMENT,
                "api_version": AZURE_API_VERSION,
            }
            # 检查是否是配置问题
            if hasattr(e, 'request'):
                error_details["request_url"] = getattr(e.request, 'url', 'N/A')
            if hasattr(e, 'response'):
                error_details["response_status"] = getattr(e.response, 'status_code', 'N/A')

            if attempt < max_retries:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries + 1}), retrying. Details: {error_details}")
                # 不等待，立即重试
            else:
                logger.error(f"Connection error after {max_retries + 1} attempts. Full details: {error_details}")
                # 如果是配置问题，给出提示
                if "YOUR ENDPOINT" in AZURE_ENDPOINT or "YOUR API KEY" in AZURE_API_KEY:
                    logger.error("⚠️  Configuration issue detected: AZURE_ENDPOINT or AZURE_API_KEY may not be set correctly!")
                return "[System] Unable to process clarification."

        except APITimeoutError as e:
            logger.error(f"Timeout error: {type(e).__name__}: {e}")
            return "[System] Unable to process clarification."

        except RateLimitError as e:
            logger.error(f"Rate limit error: {type(e).__name__}: {e}")
            return "[System] Unable to process clarification."

        except Exception as e:
            # 记录详细的异常信息，帮助诊断问题
            error_type = type(e).__name__
            error_msg = str(e)
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Unexpected error calling Azure OpenAI: {error_type}: {error_msg}")
            logger.debug(f"Full traceback: {error_traceback}")

            # 检查是否是配置问题
            if "YOUR ENDPOINT" in AZURE_ENDPOINT or "YOUR API KEY" in AZURE_API_KEY:
                logger.error("⚠️  Configuration issue detected: AZURE_ENDPOINT or AZURE_API_KEY may not be set correctly!")

            # 其他错误直接返回，不重试
            return "[System] Unable to process clarification."

    return "I need more information to answer this question."

@app.post("/generate", response_model=SingleQueryResponse)
def generate_single_response(request: SingleQueryRequest):
    """
    生成单个用户回答的API端点
    """
    try:
        # 验证必要字段
        if not request.question or not request.clarification_question:
            raise HTTPException(status_code=400, detail="Missing required fields: question, clarification_question")
        
        # 生成模拟回答
        response = simulate_user_response(request)
        
        return SingleQueryResponse(
            response=response,
            question=request.question,
            clarification_question=request.clarification_question,
            unambiguous_question=request.unambiguous_question,
            context=request.context,
            data_source=request.data_source,
            reference_question=request.reference_question,
            reference_answer=request.reference_answer,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_generate", response_model=BatchQueryResponse)
def generate_batch_response(request: BatchQueryRequest):
    """
    批量生成用户回答的API端点（模仿retrieval_server.py的方式）
    """
    try:
        queries = request.queries
        return_scores = request.return_scores
        
        if not queries:
            raise HTTPException(status_code=400, detail="No queries provided in batch request")
        
        results = []
        for idx, query in enumerate(queries):
            # 处理空 question
            if not query.question:
                if "User's actual intent:" in query.context:
                    query.question = query.context.split("User's actual intent:", 1)[1].strip().split('\n')[0].strip()
                query.question = query.question or "[Original question not provided]"
                logger.warning(f"Query has empty question field, using fallback: '{query.question}'")
            
            if not query.clarification_question:
                logger.warning(f"Skipping invalid query (missing clarification_question): {query}")
                results.append({
                    "response": "[System] Unable to process clarification.",
                    "question": query.question,
                    "clarification_question": "",
                    "context": query.context,
                    "data_source": query.data_source
                })
                continue
            
            # 批量请求时最小延迟，避免请求过快
            if idx > 0 and idx % 20 == 0:  # 每20个请求后短暂延迟
                time.sleep(0.1)  # 仅100ms延迟
            
            response = simulate_user_response(query)  # 使用默认 max_retries=1
            results.append({
                "response": response,
                "question": query.question,
                "clarification_question": query.clarification_question,
                "unambiguous_question": query.unambiguous_question,
                "context": query.context,
                "data_source": query.data_source,
                "reference_question": query.reference_question,
                "reference_answer": query.reference_answer,
            })
        
        return BatchQueryResponse(
            result=results,
            return_scores=return_scores
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clarify", response_model=BatchQueryResponse)
def clarify_alias(request: BatchQueryRequest):
    """
    兼容旧路径 /clarify，转发到 batch_generate 逻辑
    """
    return generate_batch_response(request)

@app.get("/health")
def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy", "service": "gpt-simulator"}

@app.get("/")
def root():
    """
    根端点，显示服务信息
    """
    return {
        "service": "GPT Simulator for AmbigQA",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "Generate single user response to clarification question",
            "POST /batch_generate": "Generate batch user responses (supports generation.py format)",
            "GET /health": "Health check",
            "GET /": "Service information"
        },
        "azure_config": {
            "endpoint": AZURE_ENDPOINT,
            "deployment": AZURE_DEPLOYMENT,
            "api_version": AZURE_API_VERSION
        }
    }

if __name__ == '__main__':
    # 检查必要的环境变量
    if AZURE_ENDPOINT == "YOUR ENDPOINT" or AZURE_API_KEY == "YOUR API KEY":
        logger.warning("Please set AZURE_ENDPOINT and AZURE_API_KEY environment variables")
        logger.warning("Or update the default values in the script")
    
    logger.info(f"Starting GPT Simulator service on port 8001")
    logger.info(f"Azure OpenAI Endpoint: {AZURE_ENDPOINT}")
    logger.info(f"Azure OpenAI Deployment: {AZURE_DEPLOYMENT}")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)



