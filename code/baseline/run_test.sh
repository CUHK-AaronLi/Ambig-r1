#!/bin/bash

# CoT测试快速启动脚本

echo "🚀 CoT模糊问题检测测试"
echo "========================"

# 检查是否设置了Azure配置
if [ -z "$AZURE_ENDPOINT" ] || [ -z "$AZURE_API_KEY" ]; then
    echo "⚠️  未检测到Azure OpenAI配置，正在设置..."
    source setup_azure.sh
fi

# 检查Python依赖
echo "📦 检查Python依赖..."
if ! python -c "import openai" 2>/dev/null; then
    echo "安装依赖包..."
    pip install -r requirements.txt
fi

# 运行快速测试
echo ""
echo "🧪 运行快速测试..."
python quick_cot_test.py

echo ""
echo "📊 运行演示..."
python demo.py

echo ""
echo "✅ 测试完成！"
echo ""
echo "💡 要测试特定数据集，请运行："
echo "   python cot_test_runner.py --dataset_path /path/to/dataset.json --max_samples 10"


