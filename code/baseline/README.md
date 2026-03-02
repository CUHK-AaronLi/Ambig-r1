# CoT模糊问题检测测试脚本使用说明

## 概述

这个项目提供了基于Chain-of-Thought (CoT)提示的模糊问题检测测试脚本，使用OpenAI的ChatGPT API来分析问题的模糊性。

## 文件说明

- `cot-prompt.py`: CoT提示模板和响应解析函数
- `quick_cot_test.py`: 快速测试脚本，用于验证CoT集成
- `cot_test_runner.py`: 完整的测试运行器，支持多个数据集
- `requirements.txt`: Python依赖包

## 安装依赖

```bash
pip install -r requirements.txt
```

## 设置API配置

### 方法1：使用Azure OpenAI（推荐）

```bash
# 运行配置脚本
source setup_azure.sh

# 或者手动设置环境变量
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"
```

### 方法2：使用OpenAI

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

## 使用方法

### 1. 快速测试

运行快速测试脚本验证CoT集成：

```bash
python quick_cot_test.py
```

这个脚本会：
- 测试OpenAI API连接
- 运行3个预定义的测试用例
- 测试数据集样本（如果找到的话）

### 2. 完整数据集测试

使用测试运行器对特定数据集进行测试：

```bash
python cot_test_runner.py --dataset_path /path/to/dataset.json --dataset_type auto
```

参数说明：
- `--dataset_path`: 数据集文件路径（必需）
- `--dataset_type`: 数据集类型（auto/ambignq/coqa/pacific，默认auto）
- `--model`: 模型名称（可选，优先使用环境变量）
- `--api_key`: API密钥（可选，优先使用环境变量）
- `--azure_endpoint`: Azure OpenAI端点（可选，优先使用环境变量）
- `--azure_deployment`: Azure OpenAI部署名称（可选，优先使用环境变量）
- `--azure_api_version`: Azure API版本（可选，优先使用环境变量）
- `--output`: 输出文件名（默认cot_test_results.json）
- `--max_samples`: 最大测试样本数（可选，用于快速测试）

### 3. 示例命令

```bash
# 使用Azure OpenAI测试AmbigQA数据集
python cot_test_runner.py \
    --dataset_path /mnt/users_home/cpii.local/yli/Ambig-R1/data/ambignq/dev.json \
    --dataset_type ambignq \
    --max_samples 10

# 使用Azure OpenAI测试CoQA数据集
python cot_test_runner.py \
    --dataset_path /mnt/users_home/cpii.local/yli/Ambig-R1/data/AKBC2021-Abg-CoQA/abg-coqa/coqa_abg_test.json \
    --dataset_type coqa

# 使用OpenAI测试PACIFIC数据集
export OPENAI_API_KEY='your-key'
python cot_test_runner.py \
    --dataset_path /mnt/users_home/cpii.local/yli/Ambig-R1/data/PACIFIC/data/pacific/validation.json \
    --dataset_type pacific \
    --model gpt-4
```

## 输出文件

测试完成后会生成以下文件：

1. `cot_test_results.json`: 详细的测试结果
2. `cot_test_results_summary.json`: 测试摘要统计

## CoT提示模板

脚本使用的CoT提示模板：

```
Given the document and the conversation history, first identify whether the question is ambiguous or not. If it is ambiguous, ask a clarifying question. If it is not ambiguous, answer the question. The response should start with the ambiguity analysis of the question and then follow by "Therefore, the question is not ambiguous. The answer is" or "Therefore, the question is ambiguous. The clarifying question is".

Document: {document}
Conversation History: {conversation_history}
Question: {question}

Please analyze the ambiguity of the question and provide your response:
```

## 支持的数据集格式

脚本自动检测并支持以下数据集格式：

1. **AmbigQA**: 包含`question`、`used_queries`等字段
2. **CoQA**: 包含`question`、`passage`、`answer`等字段
3. **PACIFIC**: 包含`question`、`table`等字段
4. **通用格式**: 包含`question`字段的JSON文件

## 注意事项

1. 确保设置了正确的OpenAI API密钥
2. API调用有频率限制，脚本会自动添加延迟
3. 建议先用`--max_samples`参数进行小规模测试
4. 不同模型的性能和成本不同，可根据需要选择

## 故障排除

1. **API连接失败**: 检查网络连接和API密钥
2. **数据集加载失败**: 检查文件路径和格式
3. **解析错误**: 检查CoT响应格式是否符合预期

## 扩展功能

可以通过修改`cot-prompt.py`中的提示模板来调整CoT分析的行为，或者修改`cot_test_runner.py`来支持更多数据集格式。
