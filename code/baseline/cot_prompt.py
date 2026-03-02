#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) 提示模板
用于模糊问题检测和澄清问题生成
"""

def get_cot_prompt_template():
    """
    获取CoT提示模板
    
    Returns:
        str: CoT提示模板
    """
    return """Given the document and the conversation history, first identify whether the question is ambiguous or not. If it is ambiguous, ask a clarifying question. If it is not ambiguous, answer the question. The response should start with the ambiguity analysis of the question and then follow by "Therefore, the question is not ambiguous. The answer is" or "Therefore, the question is ambiguous. The clarifying question is".

Document: {document}
Conversation History: {conversation_history}
Question: {question}

Please analyze the ambiguity of the question and provide your response:"""

def format_cot_prompt(document: str, conversation_history: str, question: str) -> str:
    """
    格式化CoT提示
    
    Args:
        document (str): 文档内容
        conversation_history (str): 对话历史
        question (str): 问题
    
    Returns:
        str: 格式化后的提示
    """
    template = get_cot_prompt_template()
    return template.format(
        document=document,
        conversation_history=conversation_history,
        question=question
    )

def parse_cot_response(response: str) -> dict:
    """
    解析CoT响应，提取模糊性判断和答案/澄清问题
    
    Args:
        response (str): LLM的响应
    
    Returns:
        dict: 包含以下字段的字典:
            - is_ambiguous: bool, 问题是否模糊
            - answer: str, 如果问题不模糊，则为答案
            - clarifying_question: str, 如果问题模糊，则为澄清问题
            - analysis: str, 模糊性分析
    """
    response = response.strip()
    
    # 初始化结果
    result = {
        'is_ambiguous': False,
        'answer': '',
        'clarifying_question': '',
        'analysis': '',
        'raw_response': response
    }
    
    # 查找模糊性分析部分
    analysis_keywords = ['ambiguity analysis', 'analysis', 'analyzing']
    analysis_start = -1
    for keyword in analysis_keywords:
        idx = response.lower().find(keyword.lower())
        if idx != -1:
            analysis_start = idx
            break
    
    # 提取分析部分
    if analysis_start != -1:
        # 找到"Therefore"之前的部分作为分析
        therefore_idx = response.find('Therefore')
        if therefore_idx != -1:
            result['analysis'] = response[analysis_start:therefore_idx].strip()
        else:
            result['analysis'] = response[analysis_start:].strip()
    
    # 判断是否模糊
    if 'not ambiguous' in response.lower():
        result['is_ambiguous'] = False
        # 提取答案
        answer_start = response.find('The answer is')
        if answer_start != -1:
            result['answer'] = response[answer_start + len('The answer is'):].strip()
    elif 'ambiguous' in response.lower() and 'not ambiguous' not in response.lower():
        result['is_ambiguous'] = True
        # 提取澄清问题
        question_start = response.find('The clarifying question is')
        if question_start != -1:
            result['clarifying_question'] = response[question_start + len('The clarifying question is'):].strip()
    
    return result

def create_test_prompt(question: str, document: str = "", conversation_history: str = "") -> str:
    """
    创建测试用的CoT提示
    
    Args:
        question (str): 测试问题
        document (str): 相关文档（可选）
        conversation_history (str): 对话历史（可选）
    
    Returns:
        str: 格式化的提示
    """
    return format_cot_prompt(
        document=document or "No specific document provided.",
        conversation_history=conversation_history or "No previous conversation.",
        question=question
    )

# 示例用法
if __name__ == "__main__":
    # 测试CoT提示模板
    test_question = "What is Python?"
    test_document = "Python is a programming language created by Guido van Rossum."
    test_history = "User: I want to learn programming."
    
    prompt = create_test_prompt(test_question, test_document, test_history)
    print("CoT提示模板:")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # 测试响应解析
    test_response = """Looking at this question, I need to analyze its ambiguity. The question "What is Python?" could refer to either the programming language or the snake species. Without additional context, this question is ambiguous because "Python" has multiple meanings.

Therefore, the question is ambiguous. The clarifying question is: Do you mean the Python programming language or the Python snake species?"""
    
    parsed = parse_cot_response(test_response)
    print("\n解析结果:")
    print(f"是否模糊: {parsed['is_ambiguous']}")
    print(f"澄清问题: {parsed['clarifying_question']}")
    print(f"分析: {parsed['analysis']}")
