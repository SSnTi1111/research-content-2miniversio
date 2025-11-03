import config
import kernels
import cuda_utils
import llm_api as agents # 用于 agents.call_llm()
import prompts            
import torch
from tqdm import tqdm
import os
import re
import ast
import sys

def extract_code(response_text):
    """(此函数保持不变)"""
    match = re.search(r'```cuda\n(.*?)```', response_text, re.DOTALL)
    if not match:
        if "torch::Tensor gemm_cuda" in response_text: 
             return response_text
        print("[Coder Agent] Error: No CUDA code block found in response.")
        return None
            
    return match.group(1).strip()

def extract_metrics(response_text):
    """(此函数保持不变)"""
    try:
        metrics_list_str = response_text.split("METRICS:")[1].strip()
        metrics_list = ast.literal_eval(metrics_list_str) 
        return metrics_list
    except Exception as e:
        print(f"[Tool Agent] Error parsing metrics list: {e}\nResponse was: {response_text}")
        return None

def main():
    print(f"Starting GEMM optimization for {config.MATRIX_N}x{config.MATRIX_N} matrix.")
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 CUDA。无法进行本地测试。")
        sys.exit(1)
        
    print(f"Running on device: {config.DEVICE}")
    print(f"Total iteration rounds: {config.ITERATION_ROUNDS}")
    if config.MOCK_LLM_CALLS:
        print("--- 警告: MOCK LLM CALLS ARE ENABLED (in config.py) ---")
        print("--- NCU 分析仍将运行，但 LLM 不会真正思考 ---")
    
    # 1. 初始化
    N = config.MATRIX_N
    device = torch.device(config.DEVICE)
    print("Initializing Tensors...")
    torch.manual_seed(42)
    A_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    B_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    print("Running PyTorch baseline (torch.matmul) for reference...")
    C_ref_torch = torch.matmul(A_torch, B_torch) 
    
    # 历史记录
    cpp_source = kernels.CPP_SOURCE 
    best_kernel_code_cuda = kernels.NAIVE_CUDA_SOURCE
    best_time_ms = float('inf')
    history = [] # (round, time_ms, kernel_code)
    
    # [!!! 已更新 !!!]
    # 'current_metrics' 现在是完整的 NCU 指标字典
    current_ncu_metrics = {} 
    current_ptxas_metrics = {}

    # 2. 获取基线性能 (Round 0)
    print("\n--- Round 0: Compiling and analyzing baseline (naive) kernel ---")
    current_kernel_code = kernels.NAIVE_CUDA_SOURCE
    current_module_name = "gemm_evolved_0"
    
    try:
        # 编译并获取 ptxas 日志
        module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
            cpp_source, 
            current_kernel_code, 
            module_name=current_module_name
        )
        print("Baseline kernel compiled successfully.")
        
        # [!!! 新增 !!!] 解析 PTXAS 指标
        current_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
        
        # 检查正确性
        is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
        if not is_correct:
            print("❌ Baseline kernel is INCORRECT. Exiting.")
            return
            
        print("Baseline kernel is correct. Benchmarking...")
        
        # [!!! 已更新 !!!] 真实评测性能
        current_time_ms = cuda_utils.benchmark_kernel(A_torch, B_torch)
        
        print("Analyzing baseline kernel with NCU (this may take a while)...")
        # [!!! 已更新 !!!] 真实 NCU 分析
        current_ncu_metrics = cuda_utils.get_real_ncu_metrics(
            module.__file__,  # 编译好的 .so 文件路径
            current_module_name,
            N
        )
        
        best_time_ms = current_time_ms
        history.append((0, best_time_ms, current_kernel_code))
        print(f"Baseline performance: {best_time_ms:.3f} ms")

    except Exception as e:
        print(f"❌ Baseline kernel failed compilation or runtime. Exiting. \n{e}")
        return

    # 3. 开始优化循环
    for i in tqdm(range(1, config.ITERATION_ROUNDS + 1), desc="Optimization Rounds"):
        print(f"\n--- Round {i}/{config.ITERATION_ROUNDS} ---")
        
        # 1. Planner Agent
        print("[Planner Agent] Analyzing kernel...")
        planner_response = agents.call_llm(
            prompts.PLANNER_SYSTEM_PROMPT,
            f"Current C++/CUDA Source:\n{current_kernel_code}"
        )
        if not planner_response or "OPTIMIZATION_GOAL:" not in planner_response:
            print("Planner failed, skipping round.")
            continue
        opt_goal = planner_response.split("OPTIMIZATION_GOAL:")[1].strip()
        print(f"[Planner Agent] Goal: {opt_goal}")
            
        # 2. Tool Agent
        print("[Tool Agent] Selecting metrics...")
        # [!!! 已更新 !!!] 向 Tool Agent 发送所有 metric *名称*
        all_metric_names = list(current_ncu_metrics.keys())
        if not all_metric_names:
            all_metric_names = config.BASE_NCU_METRICS_LIST_EXAMPLE # 后备
            
        tool_response = agents.call_llm(
            prompts.TOOL_SYSTEM_PROMPT,
            f"All Available NCU Metric Names ({len(all_metric_names)}): {all_metric_names}\n\nOptimization Goal: {opt_goal}"
        )
        relevant_metric_names = extract_metrics(tool_response)
        if not relevant_metric_names:
            print("Tool Agent failed, skipping round.")
            continue
        print(f"[Tool Agent] Selected {len(relevant_metric_names)} metrics: {relevant_metric_names}")
        
        # [!!! 已更新 !!!] 筛选 *上一次* 运行的指标
        relevant_metrics_dict = {
            metric: current_ncu_metrics.get(metric, 0.0) 
            for metric in relevant_metric_names
        }
        
        # 3. Analysis Agent
        print("[Analysis Agent] Formulating plan...")
        # [!!! 已更新 !!!] 向 Analysis Agent 发送 ptxas 和 ncu 指标
        analysis_response = agents.call_llm(
            prompts.ANALYSIS_SYSTEM_PROMPT,
            f"Current C++/CUDA Source:\n{current_kernel_code}\n\n"
            f"Optimization Goal: {opt_goal}\n\n"
            f"Current Compiler Stats: {current_ptxas_metrics}\n\n"
            f"Previous Run's Hardware Metrics: {relevant_metrics_dict}"
        )
        if not analysis_response or "DETAILED_PLAN:" not in analysis_response:
            print("Analysis Agent failed, skipping round.")
            continue
        detailed_plan = analysis_response.split("DETAILED_PLAN:")[1].strip()
        # print(f"[Analysis Agent] Plan:\n{detailed_plan}") # 计划可能很长，默认关闭

        # 4. Coder Agent
        print("[Coder Agent] Generating new kernel...")
        coder_response = agents.call_llm(
            prompts.CODER_SYSTEM_PROMPT,
            f"Original C++/CUDA Source:\n{current_kernel_code}\n\nDetailed Plan:\n{detailed_plan}"
        )
        new_kernel_code = extract_code(coder_response)
        if not new_kernel_code:
            print("Coder Agent failed to produce code, skipping round.")
            continue
        print("[Coder Agent] New kernel source generated.")
            
        # 5. 验证和分析
        try:
            current_module_name = f"gemm_evolved_{i}" 
            print(f"Compiling new kernel (module: {current_module_name})...")
            
            # [!!! 已更新 !!!] 编译并获取 PTXAS
            module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
                cpp_source, 
                new_kernel_code, 
                module_name=current_module_name
            )
            print("Compilation successful.")
            
            new_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
            
            # 检查正确性
            is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
            if not is_correct:
                print("New kernel is INCORRECT. Discarding.")
                continue
                
            print("New kernel is CORRECT. Benchmarking...")
            
            # [!!! 已更新 !!!] 真实评测和 NCU 分析
            new_time_ms = cuda_utils.benchmark_kernel(A_torch, B_torch)
            print("Analyzing new kernel with NCU...")
            new_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, 
                current_module_name, 
                N
            )
            
            # 6. 更新状态
            history.append((i, new_time_ms, new_kernel_code))
            current_kernel_code = new_kernel_code
            current_ncu_metrics = new_ncu_metrics
            current_ptxas_metrics = new_ptxas_metrics
            
            if new_time_ms < best_time_ms:
                print(f"✅ New best kernel found! Performance: {new_time_ms:.3f} ms (Old best: {best_time_ms:.3f} ms)")
                best_time_ms = new_time_ms
                best_kernel_code_cuda = new_kernel_code
            else:
                print(f"No improvement. Performance: {new_time_ms:.3f} ms (Best: {best_time_ms:.3f} ms)")

        except Exception as e:
            print(f"❌ Kernel failed compilation or runtime. Discarding iteration. Error:")
            print(e) # 详细的错误信息
            continue

    # 4. 最终报告
    print("\n--- Optimization Finished ---")
    print(f"Baseline performance (Round 0): {history[0][1]:.3f} ms")
    print(f"Best kernel performance: {best_time_ms:.3f} ms")
    
    final_kernel_path = "best_gemm_kernel.cu"
    with open(final_kernel_path, "w") as f:
        f.write(best_kernel_code_cuda)
    print(f"Best kernel C++/CUDA source saved to {final_kernel_path}")
    
    # 5. 运行最终基准测试
    print("\n--- Running Final Benchmark ---")
    pytorch_time_ms = cuda_utils.get_pytorch_performance(A_torch, B_torch)
    print(f"PyTorch (torch.matmul) performance: {pytorch_time_ms:.3f} ms")
    print(f"Our best LLM-optimized kernel: {best_time_ms:.3f} ms")
    
    speedup = pytorch_time_ms / best_time_ms
    if best_time_ms < pytorch_time_ms:
        print(f"SUCCESS: Optimized kernel is {speedup:.2f}x faster than PyTorch!")
    else:
        print(f"Result: PyTorch is {1/speedup:.2f}x faster.")

if __name__ == "__main__":
    main()