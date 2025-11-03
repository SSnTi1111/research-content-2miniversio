import torch
from torch.utils.cpp_extension import load_inline
import os
import re
import config
import time
import random

# 编译后的模块的全局缓存
_gemm_module = None

def load_gemm_module(cpp_source, cuda_source, module_name="gemm_evolved_default"):
    """
    使用PyTorch的JIT编译C++/CUDA源码。
    
    此函数使用 os.pipe/os.dup2 来捕获子进程(nvcc, ninja)
    的底层 stdout/stderr。
    
    返回 (module, stdout_log, stderr_log)
    """
    global _gemm_module
    
    # 从 CUDA 源码中提取 BLOCK_SIZE
    block_size = 16 # 默认值
    try:
        match = re.search(r'#define\s+BLOCK_SIZE\s+(\d+)', cuda_source)
        if match:
            block_size = int(match.group(1))
    except:
        pass 
        
    cuda_flags = [
        '-O3',
        '-allow-unsupported-compiler',
        f'-DBLOCK_SIZE={block_size}',
        '--ptxas-options=-v', # 请求 ptxas 详细输出 (寄存器/smem使用情况)
        '-gencode=arch=compute_80,code=sm_80' # 假设为 A100/A800。根据需要修改
    ]

    # 1. 保存原始的文件描述符
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)

    # 2. 创建管道
    r_out, w_out = os.pipe()
    r_err, w_err = os.pipe()

    # 3. 重定向 stdout/stderr
    os.dup2(w_out, 1)
    os.dup2(w_err, 2)

    # 4. 关闭原始的写入端
    os.close(w_out)
    os.close(w_err)

    stdout_log = ""
    stderr_log = ""
    _module = None

    try:
        # 5. 运行 load_inline
        _module = load_inline(
            name=module_name, # <--- 关键修改：使用唯一的名称
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["gemm_cuda"],
            verbose=True, # <--- 必须为 True 才能捕获日志
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags
        )
        
        # --- 成功路径 ---
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        stdout_bytes = os.read(r_out, 100000)
        stderr_bytes = os.read(r_err, 100000)
        stdout_log = stdout_bytes.decode('utf-8', errors='ignore')
        stderr_log = stderr_bytes.decode('utf-8', errors='ignore')
        
    except Exception as e:
        # --- 失败路径 ---
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        stdout_bytes = os.read(r_out, 100000)
        stderr_bytes = os.read(r_err, 100000)
        stdout_log = stdout_bytes.decode('utf-8', errors='ignore')
        stderr_log = stderr_bytes.decode('utf-8', errors='ignore')
        
        detailed_error_msg = f"""CUDA C++ 扩展编译失败: {e}
--- [ NVCC/Ninja STDOUT ] ---
{stdout_log}
--- [ NVCC/Ninja STDERR ] ---
{stderr_log}
-----------------------------
"""
        raise RuntimeError(detailed_error_msg)

    finally:
        # 9. (总是执行) 清理所有文件描述符
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)
        os.close(r_out)
        os.close(r_err)

    # 10. (仅成功路径到达这里) 返回模块和日志
    _gemm_module = _module
    return _gemm_module, stdout_log, stderr_log


def run_gemm(A_tensor, B_tensor):
    """
    运行已编译的GEMM模块。
    假定 A 和 B 是 PyTorch CUDA 张量。
    """
    if _gemm_module is None:
        raise RuntimeError("模块未编译。请先调用 load_gemm_module()")
    
    # 调用 C++ 扩展函数
    return _gemm_module.gemm_cuda(A_tensor, B_tensor)


def check_correctness(A_torch, B_torch, C_ref_torch):
    """
    运行已编译的内核并检查其输出的正确性。
    在内部调用 run_gemm。
    """
    print("Running evolved kernel for correctness check...")
    try:
        C_evolved = run_gemm(A_torch, B_torch)
        
        # 检查正确性
        is_correct = torch.allclose(C_evolved, C_ref_torch, atol=1e-3, rtol=1e-3)
        if not is_correct:
            print("--- KERNEL IS INCORRECT ---")
            print("Baseline [0,0]:", C_ref_torch[0,0].item())
            print("Evolved [0,0]:", C_evolved[0,0].item())
            print("---------------------------")
        return is_correct

    except Exception as e:
        print(f"--- KERNEL RUNTIME FAILED ---")
        print(e)
        print("-----------------------------")
        return False


def get_all_ncu_metrics(kernel_code, N):
    """
    MOCK FUNCTION: 模拟运行 NCU 并获取全部27个指标。
    (这部分保持不变，它模拟的是 *运行时* 分析)
    """
    print("Simulating NCU profiling...")
    
    mock_metrics = {}
    for metric in config.ALL_NCU_METRICS_LIST:
        if "bytes" in metric or "transactions" in metric:
            mock_metrics[metric] = random.uniform(1e7, 1e9)
        elif "efficiency" in metric or "occupancy" in metric:
            mock_metrics[metric] = random.uniform(50.0, 95.0)
        else:
            mock_metrics[metric] = random.uniform(1e4, 1e7)
    
    if "__shared__" not in kernel_code: # 朴素Kernel
        time_ms = random.uniform(100, 150)
        mock_metrics["dram__bytes_read.sum"] = random.uniform(1e9, 2e9)
    else: # 优化的Kernel
        time_ms = random.uniform(20, 40)
        mock_metrics["dram__bytes_read.sum"] = random.uniform(1e8, 5e8)
        mock_metrics["shared_load_transactions.sum"] = random.uniform(1e8, 1e9)

    print(f"Mocked performance: {time_ms:.3f} ms")
    return mock_metrics, time_ms


def get_pytorch_performance(A_torch, B_torch):
    """获取PyTorch的基准性能。(不变)"""
    print("Warming up PyTorch...")
    for _ in range(10):
        _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(50):
        _ = torch.matmul(A_torch, B_torch)
    end.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start.elapsed_time(end) / 50
    return avg_time_ms