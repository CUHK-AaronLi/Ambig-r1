#!/bin/bash

# GPT Simulator启动脚本

echo "Starting GPT Simulator Service..."

# 设置Azure OpenAI环境变量
export AZURE_ENDPOINT="https://cpii-s5.openai.azure.com/"
export AZURE_API_KEY="91e5ea9bf61c4769a44b0b0b5c67d559"
export AZURE_DEPLOYMENT="gpt-4o"
export AZURE_API_VERSION="2024-02-01"

# 检查环境变量是否设置
if [ "$AZURE_ENDPOINT" = "YOUR ENDPOINT" ] || [ "$AZURE_API_KEY" = "YOUR API KEY" ]; then
    echo "Warning: Please set AZURE_ENDPOINT and AZURE_API_KEY environment variables"
    echo "Or update the values in this script"
    echo ""
    echo "Example:"
    echo "export AZURE_ENDPOINT='https://your-resource.openai.azure.com/'"
    echo "export AZURE_API_KEY='your-api-key-here'"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# 安装依赖
# echo "Installing dependencies..."
#pip install -r requirements_gpt_simulator.txt

# 启动服务
echo "Starting service on port 8001..."
python gpt_simulator.py






