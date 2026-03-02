# AmbigQA 数据处理脚本使用说明

## 概述

这个脚本用于将AmbigQA数据集转换为VERL框架训练所需的parquet格式。AmbigQA是一个处理模糊问题的数据集，每个问题可能有多个正确答案，需要模型能够主动澄清问题。

**注意**: 本脚本使用 `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'ambig_qa')` 直接加载预处理好的数据集。

## 功能特点

### 🎯 **多动作支持**
- `<search>`: 搜索更多信息
- `<clarify>`: 澄清模糊问题
- `<answer>`: 提供最终答案

### 📊 **数据处理**
- 直接从Hugging Face加载AmbigQA数据集
- 自动应用提示模板
- 生成训练集和测试集
- 支持HDFS存储

### 🔧 **模板系统**
- `base`: 支持所有三种动作的完整模板

## 使用方法

### 基本用法

```bash
python ambignq.py --local_dir ./data/ambignq_search
```

### 完整参数

```bash
python ambignq.py \
    --local_dir ./data/ambignq_search \
    --hdfs_dir /hdfs/path/ambignq \
    --template_type base
```

### 参数说明

- `--local_dir`: 本地输出目录（默认: `./data/ambignq_search`）
- `--hdfs_dir`: HDFS输出目录（可选）
- `--template_type`: 提示模板类型（默认: `base`）

## 数据来源

脚本使用以下数据源：
```python
dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'ambig_qa')
```

这个数据集已经预处理好了AmbigQA数据，包含：
- `train`: 训练集
- `test`: 测试集

## 提示模板

### Base模板

```
Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, you can choose one of three actions:
1. If you need more information, use <search> query </search> to search and get results between <information> and </information>.
2. If the question is ambiguous, use <clarify> clarification question </clarify> to ask for clarification.
3. If you have enough information, provide the answer inside <answer> and </answer>.

You can search or clarify as many times as needed but only when necessary. Question: {question}
```

## 输出文件

脚本会生成以下文件：

- `train.parquet`: 训练集数据
- `test.parquet`: 测试集数据

## 数据格式

### 输出格式

处理后的数据包含以下结构：

```json
{
    "data_source": "ambignq",
    "prompt": [{
        "role": "user",
        "content": "Answer the given question... Question: What is the capital of France?\n"
    }],
    "ability": "ambiguous-question-answering",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "target": ["Paris"]
        }
    },
    "extra_info": {
        "split": "train",
        "index": 0
    }
}
```

## 示例

### 处理AmbigQA数据

```bash
# 基本处理
python ambignq.py --local_dir ./data/ambignq_search

# 同时保存到HDFS
python ambignq.py \
    --local_dir ./data/ambignq_search \
    --hdfs_dir /hdfs/user/ambignq
```

## 测试

运行测试脚本验证功能：

```bash
python test_ambignq.py
```

测试内容包括：
- 提示模板生成
- 数据集加载
- 数据处理流程
- 模板变体测试

## 注意事项

1. **网络连接**: 需要网络连接来下载Hugging Face数据集
2. **输出目录**: 脚本会自动创建输出目录
3. **HDFS**: 如果使用HDFS，确保有相应的权限
4. **依赖**: 需要安装 `datasets` 库

## 故障排除

### 常见错误

1. **网络错误**: 检查网络连接和防火墙设置
2. **权限错误**: 确保有读写权限
3. **依赖缺失**: 安装必要的Python包

### 调试模式

如果遇到问题，可以：
1. 运行测试脚本检查基本功能
2. 检查网络连接
3. 验证Python环境

## 与VERL框架集成

处理后的数据可以直接用于VERL框架训练：

1. 在训练配置中指定数据路径
2. 使用`data_source: "ambignq"`标识数据源
3. 配置相应的奖励函数和训练参数

## 依赖要求

```bash
pip install datasets
pip install pandas
pip install pyarrow  # 用于parquet格式
```
