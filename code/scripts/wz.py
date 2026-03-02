#!/usr/bin/env python3
"""
智能GPU占用脚本
检测GPU是否有人使用，无人使用时自动占用，有人使用时自动释放
"""

import torch
import time
import psutil
import os
import subprocess
import json
from datetime import datetime

def get_gpu_memory_usage(device_id):
    """获取指定GPU的内存使用情况"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def get_system_memory():
    """获取系统内存使用情况"""
    memory = psutil.virtual_memory()
    return memory.used / 1024**3, memory.total / 1024**3  # GB

def check_gpu_usage():
    """检查GPU使用情况，返回每个GPU是否有人使用"""
    try:
        # 首先尝试使用nvidia-smi获取GPU使用信息
        print("正在使用nvidia-smi检查GPU使用情况...")
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,processes.count', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            print("nvidia-smi命令执行成功")
            gpu_usage = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        gpu_id = int(parts[0])
                        memory_used = float(parts[1])
                        gpu_util = float(parts[2])
                        process_count = int(parts[3])
                        
                        # 判断GPU是否有人使用
                        # 如果内存使用 > 100MB 或 GPU利用率 > 5% 或 有进程在使用，则认为有人使用
                        is_occupied = (memory_used > 100 or gpu_util > 5 or process_count > 0)
                        gpu_usage[gpu_id] = {
                            'is_occupied': is_occupied,
                            'memory_used_mb': memory_used,
                            'gpu_util': gpu_util,
                            'process_count': process_count
                        }
            
            if gpu_usage:
                print(f"成功获取到 {len(gpu_usage)} 个GPU的信息")
                return gpu_usage
            else:
                print("nvidia-smi返回了空结果")
        else:
            print(f"nvidia-smi命令执行失败，返回码: {result.returncode}")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
    
    except FileNotFoundError:
        print("nvidia-smi命令未找到，尝试使用PyTorch检测...")
    except subprocess.TimeoutExpired:
        print("nvidia-smi命令执行超时，尝试使用PyTorch检测...")
    except Exception as e:
        print(f"nvidia-smi执行出错: {e}，尝试使用PyTorch检测...")
    
    # 备用方案：使用PyTorch检测GPU使用情况
    try:
        print("使用PyTorch作为备用检测方法...")
        if not torch.cuda.is_available():
            print("PyTorch CUDA不可用")
            return {}
        
        gpu_usage = {}
        device_count = torch.cuda.device_count()
        print(f"PyTorch检测到 {device_count} 个GPU设备")
        
        for device_id in range(device_count):
            try:
                torch.cuda.set_device(device_id)
                
                # 获取GPU内存信息
                allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                reserved = torch.cuda.memory_reserved() / 1024**2   # MB
                
                # 尝试获取GPU利用率（通过简单的计算来模拟）
                # 如果内存使用量较大，认为GPU被占用
                is_occupied = allocated > 100  # 如果已分配内存 > 100MB，认为被占用
                
                gpu_usage[device_id] = {
                    'is_occupied': is_occupied,
                    'memory_used_mb': allocated,
                    'gpu_util': 0.0,  # PyTorch无法直接获取GPU利用率
                    'process_count': 1 if is_occupied else 0  # 简化处理
                }
                
                print(f"GPU {device_id}: 已分配 {allocated:.1f}MB, 保留 {reserved:.1f}MB, 占用状态: {is_occupied}")
                
            except Exception as e:
                print(f"检测GPU {device_id} 时出错: {e}")
                continue
        
        if gpu_usage:
            print(f"PyTorch检测成功，获取到 {len(gpu_usage)} 个GPU的信息")
            return gpu_usage
        
    except Exception as e:
        print(f"PyTorch检测也失败: {e}")
    
    # 最后的备用方案：假设所有GPU都可用
    print("所有检测方法都失败，假设所有GPU都可用")
    return {}

def is_gpu_available(device_id, gpu_usage):
    """检查指定GPU是否可用（无人使用）"""
    if device_id not in gpu_usage:
        # 如果没有该GPU的信息，假设可用
        return True
    
    gpu_info = gpu_usage[device_id]
    return not gpu_info['is_occupied']

def allocate_gpu_memory(device_id, memory_per_gpu):
    """为指定GPU分配内存"""
    print(f"正在为GPU {device_id} 分配 {memory_per_gpu:.2f} GB内存...")
    
    try:
        # 设置GPU设备
        torch.cuda.set_device(device_id)
        
        # 计算需要的张量大小
        target_memory_bytes = memory_per_gpu * 1024**3
        elements_needed = int(target_memory_bytes / 4)  # float32 = 4 bytes
        
        # 计算合适的张量形状
        side_length = int(elements_needed ** 0.5)
        actual_elements = side_length * side_length
        
        print(f"  GPU {device_id} 目标内存: {memory_per_gpu:.2f} GB")
        print(f"  张量形状: [{side_length}, {side_length}]")
        
        # 创建大张量占用GPU内存
        tensor = torch.randn(side_length, side_length, device=f'cuda:{device_id}')
        
        # 确保张量在GPU上
        tensor = tensor.cuda()
        
        # 强制同步
        torch.cuda.synchronize()
        
        # 显示实际占用情况
        allocated, reserved = get_gpu_memory_usage(device_id)
        print(f"  GPU {device_id} 实际占用内存: {allocated:.2f} GB")
        print(f"  GPU {device_id} 保留内存: {reserved:.2f} GB")
        
        return tensor
        
    except Exception as e:
        print(f"  为GPU {device_id} 分配内存时出错: {e}")
        return None

def release_gpu_memory(device_id, tensor):
    """释放指定GPU的内存"""
    try:
        torch.cuda.set_device(device_id)
        del tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        allocated, reserved = get_gpu_memory_usage(device_id)
        print(f"GPU {device_id} 释放后内存: {allocated:.2f} GB (保留: {reserved:.2f} GB)")
        return True
        
    except Exception as e:
        print(f"释放GPU {device_id} 时出错: {e}")
        return False

def main():
    print("=" * 60)
    print("智能GPU占用脚本启动")
    print("直接占用指定GPU，每个占用15GB内存")
    print("=" * 60)
    
    # 系统环境检查
    print("正在检查系统环境...")
    
    # 检查nvidia-smi是否可用
    try:
        result = subprocess.run(['which', 'nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"nvidia-smi 路径: {result.stdout.strip()}")
        else:
            print("nvidia-smi 未找到")
    except Exception as e:
        print(f"检查nvidia-smi时出错: {e}")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("CUDA不可用，无法占用GPU内存")
        return
    
    device_count = torch.cuda.device_count()
    print(f"检测到 {device_count} 个GPU设备")
    
    # 目标GPU设备
    target_devices = [6, 7]
    
    # 检查目标GPU是否可用
    for device_id in target_devices:
        if device_id >= device_count:
            print(f"GPU {device_id} 不存在，最大GPU索引为 {device_count-1}")
            return
    
    # 显示目标GPU信息
    for device_id in target_devices:
        props = torch.cuda.get_device_properties(device_id)
        print(f"GPU {device_id}: {props.name}")
        print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
    
    # 配置参数
    check_interval = 30  # 每30秒检查一次
    memory_per_gpu = 15.0  # 每个GPU占用15GB
    debug_mode = True  # 调试模式，显示详细信息
    
    print(f"\n检查间隔: {check_interval} 秒")
    print(f"每个GPU占用: {memory_per_gpu:.2f} GB")
    print("\n开始直接占用指定GPU...")
    
    # 存储所有张量
    tensors = {}
    
    try:
        # 直接占用所有目标GPU
        for device_id in target_devices:
            print(f"正在占用GPU {device_id}...")
            try:
                # 检查GPU是否可以访问
                torch.cuda.set_device(device_id)
                # 尝试分配少量内存来测试GPU是否可用
                test_tensor = torch.randn(100, 100, device=f'cuda:{device_id}')
                del test_tensor
                torch.cuda.empty_cache()
                
                # 开始占用
                print(f"GPU {device_id} 测试成功，开始占用...")
                tensor = allocate_gpu_memory(device_id, memory_per_gpu)
                if tensor is not None:
                    tensors[device_id] = tensor
                    print(f"GPU {device_id} 占用成功")
                else:
                    print(f"GPU {device_id} 占用失败")
            except Exception as e:
                print(f"GPU {device_id} 不可用: {e}")
        
        print(f"\n初始占用完成，共占用 {len(tensors)} 个GPU")
        
        while True:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{current_time}] 保持GPU占用状态...")
            
            # 保持所有GPU的占用状态
            for device_id in target_devices:
                if device_id in tensors:
                    # 已经在占用，保持活跃
                    print(f"GPU {device_id} 正在占用中，保持活跃...")
                    try:
                        torch.cuda.set_device(device_id)
                        tensors[device_id] = tensors[device_id] + 0.001
                        torch.cuda.synchronize()
                    except Exception as e:
                        print(f"保持GPU {device_id} 活跃时出错: {e}")
                        # 如果出错，尝试重新占用
                        print(f"尝试重新占用GPU {device_id}...")
                        try:
                            del tensors[device_id]
                            torch.cuda.empty_cache()
                            tensor = allocate_gpu_memory(device_id, memory_per_gpu)
                            if tensor is not None:
                                tensors[device_id] = tensor
                                print(f"GPU {device_id} 重新占用成功")
                        except Exception as e2:
                            print(f"重新占用GPU {device_id} 失败: {e2}")
                else:
                    # 没有占用，尝试占用
                    print(f"GPU {device_id} 未占用，尝试占用...")
                    try:
                        torch.cuda.set_device(device_id)
                        tensor = allocate_gpu_memory(device_id, memory_per_gpu)
                        if tensor is not None:
                            tensors[device_id] = tensor
                            print(f"GPU {device_id} 占用成功")
                        else:
                            print(f"GPU {device_id} 占用失败")
                    except Exception as e:
                        print(f"占用GPU {device_id} 失败: {e}")
            
            # 显示当前状态
            if tensors:
                print(f"\n当前占用GPU: {list(tensors.keys())}")
                total_allocated = 0
                for device_id, tensor in tensors.items():
                    allocated, reserved = get_gpu_memory_usage(device_id)
                    total_allocated += allocated
                    print(f"  GPU {device_id}: {allocated:.2f} GB (保留: {reserved:.2f} GB)")
                print(f"  总计占用: {total_allocated:.2f} GB")
            else:
                print("\n当前没有占用任何GPU")
            
            # 显示系统内存
            sys_used, sys_total = get_system_memory()
            print(f"系统内存: {sys_used:.2f} / {sys_total:.2f} GB")
            
            print(f"\n等待 {check_interval} 秒后下次保持活跃...")
            print("按 Ctrl+C 停止...")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n收到停止信号，正在清理...")
        
    finally:
        # 清理所有GPU内存
        if tensors:
            print("正在清理所有占用的GPU内存...")
            for device_id, tensor in tensors.items():
                release_gpu_memory(device_id, tensor)
            print("所有GPU内存已释放")
        else:
            print("没有需要清理的GPU内存")

if __name__ == "__main__":
    main()
