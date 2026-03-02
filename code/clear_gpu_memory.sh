#!/bin/bash
# GPU内存清理脚本

echo "🧹 清理GPU内存..."

# 清理所有GPU内存
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        print(f'GPU {i} 内存已清理')
    print('所有GPU内存清理完成')
else:
    print('CUDA不可用')
"

# 显示GPU状态
echo "📊 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader,nounits

echo "✅ 内存清理完成"

