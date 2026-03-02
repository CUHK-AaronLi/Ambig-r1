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
    Create prompt for simulating user responses
    """
    ambig_q = query.question.strip()
    unambig_q = (query.unambiguous_question or query.reference_question or "").strip()
    clarif_q = query.clarification_question.strip()
    context_block = query.context.strip()

    prompt = f"""You are a helpful agent whose goal is to clarify a potentially ambiguous question. You have access to a question in two
forms. One is a potentially ambiguous form and the other is an unambiguous form of that question.
You will also be given a clarification question issued by a different agent that only has access to the ambiguous question
and is trying to clarify it. Your only job is to answer the clarification question to the best of your ability. It is
possible that the ambiguous and unambiguous questions are identical.
You are not allowed to directly reveal information about the unambiguous question. You are only allowed to answer the
clarification question based on the unambiguous version of the question. If the clarification question is not answerable
from the unambiguous question, you should answer that you do not know the answer. Again, **NEVER** just repeat the
unambiguous question as an answer. In addition, **NEVER** clarify the question using information that is not inferred
from the unambiguous question. You will get a prize if you manage to complete the task successfully. Always respond in a
single line with the format
'ACTION : **ANSWER_CLARIFICATION** : <the answer to the clarification question>'
Now for the real data:
Potentially ambiguous question: {ambig_q}
Unambiguous question: {unambig_q}
Clarification question: {clarif_q}"""

    # 如有额外上下文，附加在末尾
    if context_block:
        prompt += f"\nContext (for your reference only, do not leak it): {context_block}"

    return prompt

def simulate_user_response(query: ClarifyQuery, max_retries: int = 1) -> str:
    """
    使用Azure OpenAI模拟用户回答clarification问题，带重试机制
    """
    # 输入验证
    if not query.question or not query.clarification_question:
        logger.warning("Missing required parameters: question or clarification_question")
        return "I need more information to answer this question."

    prompt = create_simulation_prompt(query)

    # 只重试一次，快速失败
    for attempt in range(max_retries + 1):  # 0次重试 = 尝试1次，1次重试 = 尝试2次
        try:
            completion = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.7,
            )

            response = completion.choices[0].message.content.strip()
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
                return "I need more information to answer this question."

        except APITimeoutError as e:
            logger.error(f"Timeout error: {type(e).__name__}: {e}")
            return "I need more information to answer this question."

        except RateLimitError as e:
            logger.error(f"Rate limit error: {type(e).__name__}: {e}")
            return "I need more information to answer this question."

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
            return "I need more information to answer this question."

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
                    "response": "I need more information to answer this question.",
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



