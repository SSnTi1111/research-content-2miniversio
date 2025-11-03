import config
import kernels
import cuda_utils
import llm_api as agents # 用于 agents.call_llm()
import prompts            # <--- [修复 1] 添加此导入
import torch
from tqdm import tqdm
import os
import re
import ast
import sys

def extract_code(response_text):
    """从LLM的回复中提取CUDA代码块。"""
    match = re.search(r'```cuda\n(.*?)```', response_text, re.DOTALL)
    if not match:
        if "torch::Tensor gemm_cuda" in response_text: # 备用方案
             return response_text
        print("[Coder Agent] Error: No CUDA code block found in response.")
        return None
            
    return match.group(1).strip()

def extract_metrics(response_text):
    """从LLM的回复中提取指标列表。"""
    try:
        metrics_list_str = response_text.split("METRICS:")[1].strip()
        metrics_list = ast.literal_eval(metrics_list_str) # Safely evaluate string as list
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
        print("--- MOCK LLM CALLS ARE ENABLED (in config.py) ---")
    
    # 1. 初始化 (使用 Torch)
    N = config.MATRIX_N
    device = torch.device(config.DEVICE)
    
    print("Initializing Tensors...")
    torch.manual_seed(42)
    A_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    B_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    
    print("Running PyTorch baseline (torch.matmul) for reference...")
    C_ref_torch = torch.matmul(A_torch, B_torch) # 基准答案
    
    # 历史记录
    cpp_source = kernels.CPP_SOURCE # C++ 签名是恒定的
    best_kernel_code_cuda = kernels.NAIVE_CUDA_SOURCE
    best_time_ms = float('inf')
    history = [] # (round, time_ms, kernel_code)

    # 2. 获取基线性能
    print("\n--- Round 0: Compiling and checking baseline (naive) kernel ---")
    current_kernel_code = kernels.NAIVE_CUDA_SOURCE
    
    try:
        # 编译
        _, stdout_log, stderr_log = cuda_utils.load_gemm_module(
            cpp_source, 
            current_kernel_code, 
            module_name="gemm_evolved_0"
        )
        print("Baseline kernel compiled successfully.")
        # print(stdout_log) # 取消注释以查看 ptxas 日志
        
        # 检查正确性
        is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
        if not is_correct:
            print("❌ Baseline kernel is INCORRECT. Exiting.")
            return
            
        print("Baseline kernel is correct. Profiling (mock)...")
        current_metrics, current_time_ms = cuda_utils.get_all_ncu_metrics(current_kernel_code, N)
        
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
            prompts.PLANNER_SYSTEM_PROMPT,  # <--- [修复 2] 使用 'prompts.'
            f"Current C++/CUDA Source:\n{current_kernel_code}"
        )
        if not planner_response or "OPTIMIZATION_GOAL:" not in planner_response:
            print("Planner failed, skipping round.")
            continue
        opt_goal = planner_response.split("OPTIMIZATION_GOAL:")[1].strip()
        print(f"[Planner Agent] Goal: {opt_goal}")
            
        # 2. Tool Agent
        print("[Tool Agent] Selecting metrics...")
        tool_response = agents.call_llm(
            prompts.TOOL_SYSTEM_PROMPT, # <--- [修复 2] 使用 'prompts.'
            f"All Metrics: {config.ALL_NCU_METRICS_LIST}\nOptimization Goal: {opt_goal}"
        )
        relevant_metric_names = extract_metrics(tool_response)
        if not relevant_metric_names:
            print("Tool Agent failed, skipping round.")
            continue
        print(f"[Tool Agent] Selected {len(relevant_metric_names)} metrics: {relevant_metric_names}")
        
        relevant_metrics_dict = {
            metric: current_metrics.get(metric, 0.0) 
            for metric in relevant_metric_names
        }
        
        # 3. Analysis Agent
        print("[Analysis Agent] Formulating plan...")
        analysis_response = agents.call_llm(
            prompts.ANALYSIS_SYSTEM_PROMPT, # <--- [修复 2] 使用 'prompts.'
            f"Current C++/CUDA Source:\n{current_kernel_code}\n\nOptimization Goal: {opt_goal}\n\nPrevious Metrics: {relevant_metrics_dict}"
        )
        if not analysis_response or "DETAILED_PLAN:" not in analysis_response:
            print("Analysis Agent failed, skipping round.")
            continue
        detailed_plan = analysis_response.split("DETAILED_PLAN:")[1].strip()
        print(f"[Analysis Agent] Plan:\n{detailed_plan}")
            
        # 4. Coder Agent
        print("[Coder Agent] Generating new kernel...")
        coder_response = agents.call_llm(
            prompts.CODER_SYSTEM_PROMPT, # <--- [修复 2] 使用 'prompts.'
            f"Original C++/CUDA Source:\n{current_kernel_code}\n\nDetailed Plan:\n{detailed_plan}"
        )
        new_kernel_code = extract_code(coder_response)
        if not new_kernel_code:
            print("Coder Agent failed to produce code, skipping round.")
            continue
        print("[Coder Agent] New kernel source generated.")
            
        # 5. 验证和分析 (使用 try/except 捕获编译和运行时错误)
        try:
            # 必须使用唯一的模块名
            module_name = f"gemm_evolved_{i}" 
            print(f"Compiling new kernel (module: {module_name})...")
            
            # 编译 (会捕获编译错误)
            _, stdout_log, stderr_log = cuda_utils.load_gemm_module(
                cpp_source, 
                new_kernel_code, 
                module_name=module_name
            )
            print("Compilation successful. Checking correctness...")
            
            # 检查正确性 (会捕获运行时错误)
            is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
            if not is_correct:
                print("New kernel is INCORRECT. Discarding.")
                continue
                
            print("New kernel is CORRECT. Profiling (mock)...")
            new_metrics, new_time_ms = cuda_utils.get_all_ncu_metrics(new_kernel_code, N)
            
            # 6. 更新状态
            history.append((i, new_time_ms, new_kernel_code))
            current_kernel_code = new_kernel_code
            current_metrics = new_metrics
            
            if new_time_ms < best_time_ms:
                print(f"✅ New best kernel found! Performance: {new_time_ms:.3f} ms (Old best: {best_time_ms:.3f} ms)")
                best_time_ms = new_time_ms
                best_kernel_code_cuda = new_kernel_code
            else:
                print(f"No improvement. Performance: {new_time_ms:.3f} ms (Best: {best_time_ms:.3f} ms)")

        except Exception as e:
            print(f"❌ Kernel failed compilation or runtime. Discarding iteration. Error:")
            print(e) # 详细的错误信息 (包括 stdout/stderr) 已在 cuda_utils 中被抛出
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
    print(f"Our best LLM-optimized kernel: {best_time_ms:.3f} ms (Mocked)")
    
    if best_time_ms < pytorch_time_ms:
        print("SUCCESS: Optimized kernel is faster than PyTorch!")
    else:
        print("Result: PyTorch is faster (Note: our performance is mocked).")

if __name__ == "__main__":
    main()