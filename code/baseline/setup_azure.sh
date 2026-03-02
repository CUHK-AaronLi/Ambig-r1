#!/bin/bash

# Azure OpenAI配置脚本

echo "🔧 设置Azure OpenAI环境变量..."

# 设置Azure OpenAI环境变量
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"

echo "✅ Azure OpenAI环境变量已设置:"
echo "   AZURE_ENDPOINT: $AZURE_ENDPOINT"
echo "   AZURE_DEPLOYMENT: $AZURE_DEPLOYMENT"
echo "   AZURE_API_VERSION: $AZURE_API_VERSION"
echo "   AZURE_API_KEY: ${AZURE_API_KEY:0:8}..."

echo ""
echo "🚀 现在可以运行CoT测试脚本了:"
echo "   python quick_cot_test.py"
echo "   python cot_test_runner.py --dataset_path /path/to/dataset.json"

