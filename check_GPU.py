import torch

# 检查CUDA是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA可用: {is_cuda_available}")

# 如果CUDA可用，获取GPU设备数量和名称
if is_cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数量: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i} 名称: {torch.cuda.get_device_name(i)}")