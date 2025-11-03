import config
import kernels
import cuda_utils
import llm_api as agents 
import prompts            
import torch
from tqdm import tqdm
import os
import re
import ast
import sys
import json 

def extract_code(response_text):
    """(此函数保持不变)"""
    if not response_text: return None 
    match = re.search(r'```cuda\n(.*?)```', response_text, re.DOTALL)
    if not match:
        if "torch::Tensor gemm_cuda" in response_text: 
             return response_text
        print("[Coder Agent] Error: No CUDA code block found in response.")
        return None
            
    return match.group(1).strip()

def extract_metrics(response_text):
    """(此函数保持不变)"""
    if not response_text: return None 
    try:
        metrics_list_str = response_text.split("METRICS:")[1].strip()
        metrics_list = ast.literal_eval(metrics_list_str) 
        return metrics_list
    except Exception as e:
        print(f"[Tool Agent] Error parsing metrics list: {e}\nResponse was: {response_text}")
        return None

# [!!! 已更新 !!!] 解决问题 2: TypeError
def summarize_history(history: list) -> str:
    """将优化历史转换为 LLM 可读的摘要。"""
    if not history:
        return "No previous attempts."
    
    summary = "Previous Optimization Attempts:\n"
    for i, entry in enumerate(history):
        summary += f"  Round {entry['round']}:\n"
        summary += f"    Goal: {entry['goal']}\n"
        summary += f"    Status: {entry['status']}\n"
        
        # [!!! 修复 2 !!!] 检查 time_ms 是否为 None
        perf_str = "N/A"
        if entry['time_ms'] is not None:
            perf_str = f"{entry['time_ms']:.3f} ms"
        summary += f"    Performance: {perf_str}\n"
        # [!!! 修复 2 结束 !!!]

        if entry['status'] == "Success (New Best)": # 仅在新最佳时显示指标
            summary += f"    Registers: {entry['ptxas_metrics'].get('registers_used', 'N/A')}\n"
            summary += f"    Shared Mem: {entry['ptxas_metrics'].get('shared_mem_bytes', 'N/A')} bytes\n"
        elif "Error" in entry['status'] or "Failed" in entry['status']:
            # 截断长的错误日志
            details = entry.get('details', 'No details')
            if len(details) > 200:
                details = details[:200] + "..."
            summary += f"    Error Details: {details}\n"
    return summary

def main():
    print(f"Starting GEMM optimization for {config.MATRIX_N}x{config.MATRIX_N} matrix.")
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 CUDA。无法进行本地测试。")
        sys.exit(1)
        
    print(f"Running on device: {config.DEVICE}")
    print(f"Total iteration rounds: {config.ITERATION_ROUNDS}")
    if config.MOCK_LLM_CALLS:
        print("--- 警告: MOCK LLM CALLS ARE ENABLED (in config.py) ---")
    
    # 1. 初始化
    N = config.MATRIX_N
    device = torch.device(config.DEVICE)
    print("Initializing Tensors...")
    torch.manual_seed(42)
    A_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    B_torch = torch.randn((N, N), dtype=torch.float32, device=device)
    print("Running PyTorch baseline (torch.matmul) for reference...")
    C_ref_torch = torch.matmul(A_torch, B_torch) 
    
    cpp_source = kernels.CPP_SOURCE 
    best_kernel_code_cuda = kernels.NAIVE_CUDA_SOURCE
    best_time_ms = float('inf')
    best_ptxas_metrics = {}
    best_ncu_metrics = {}
    current_ncu_metrics = {}
    
    optimization_history = []
    if os.path.exists(config.HISTORY_FILE):
        print(f"Loading existing history from {config.HISTORY_FILE}")
        with open(config.HISTORY_FILE, 'r') as f:
            optimization_history = json.load(f)

    # 2. 获取基线性能 (Round 0)
    print("\n--- Round 0: Compiling and analyzing baseline (naive) kernel ---")
    current_module_name = "gemm_evolved_0"
    
    try:
        module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
            cpp_source, 
            best_kernel_code_cuda, 
            module_name=current_module_name
        )
        print("Baseline kernel compiled successfully.")
        
        best_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
        
        is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
        if not is_correct:
            print("❌ Baseline kernel is INCORRECT. Exiting.")
            return
            
        print("Baseline kernel is correct. Benchmarking...")
        best_time_ms = cuda_utils.benchmark_kernel(A_torch, B_torch)
        
        print("Analyzing baseline kernel with NCU (this may take a while)...")
        best_ncu_metrics = cuda_utils.get_real_ncu_metrics(
            module.__file__,
            current_module_name,
            N
        )
        current_ncu_metrics = best_ncu_metrics 
        
        if not any(h['round'] == 0 for h in optimization_history):
            history_entry = {
                "round": 0,
                "goal": "Baseline",
                "status": "Success",
                "time_ms": best_time_ms,
                "ptxas_metrics": best_ptxas_metrics,
                "details": "Initial baseline measurement"
            }
            optimization_history.append(history_entry)
        
        print(f"Baseline performance: {best_time_ms:.3f} ms")

    except Exception as e:
        print(f"❌ Baseline kernel failed compilation or runtime. Exiting. \n{e}")
        return

    # 3. 开始优化循环
    for i in tqdm(range(1, config.ITERATION_ROUNDS + 1), desc="Optimization Rounds"):
        print(f"\n--- Round {i}/{config.ITERATION_ROUNDS} ---")
        
        history_summary = summarize_history(optimization_history)
        
        opt_goal = "N/A"
        detailed_plan = "N/A"
        new_kernel_code = None
        status = "Failed (Unknown)"
        details = ""
        new_time_ms = float('inf')
        new_ptxas_metrics = {}
        new_ncu_metrics = {}
        
        try:
            # 1. Planner Agent
            print("[Planner Agent] Analyzing kernel...")
            planner_response = agents.call_llm(
                "planner", 
                prompts.PLANNER_SYSTEM_PROMPT,
                f"Optimization History:\n{history_summary}\n\n"
                f"Current Best C++/CUDA Source (Time: {best_time_ms:.3f} ms):\n{best_kernel_code_cuda}"
            )
            if not planner_response or "OPTIMIZATION_GOAL:" not in planner_response:
                status, details = "Failed (Planner)", "Planner did not return a valid goal."
                print(f"❌ {status} {details}")
                continue 
            opt_goal = planner_response.split("OPTIMIZATION_GOAL:")[1].strip()
            print(f"[Planner Agent] Goal: {opt_goal}")
                
            # 2. Tool Agent
            print("[Tool Agent] Selecting metrics...")
            all_metric_names = list(current_ncu_metrics.keys())
            if not all_metric_names:
                all_metric_names = config.BASE_NCU_METRICS_LIST_EXAMPLE
                
            tool_response = agents.call_llm(
                "tool", 
                prompts.TOOL_SYSTEM_PROMPT,
                f"All Available NCU Metric Names ({len(all_metric_names)}): {all_metric_names}\n\nOptimization Goal: {opt_goal}"
            )
            relevant_metric_names = extract_metrics(tool_response)
            if not relevant_metric_names:
                status, details = "Failed (Tool)", "Tool Agent did not return a valid metric list."
                print(f"❌ {status} {details}")
                continue 
            print(f"[Tool Agent] Selected {len(relevant_metric_names)} metrics: {relevant_metric_names}")
            
            relevant_metrics_dict = {
                metric: current_ncu_metrics.get(metric, 0.0) 
                for metric in relevant_metric_names
            }
            
            # 3. Analysis Agent
            print("[Analysis Agent] Formulating plan...")
            analysis_response = agents.call_llm(
                "analysis", 
                prompts.ANALYSIS_SYSTEM_PROMPT,
                f"Optimization History:\n{history_summary}\n\n"
                f"Current Best C++/CUDA Source:\n{best_kernel_code_cuda}\n\n"
                f"Optimization Goal: {opt_goal}\n\n"
                f"Current Best Compiler Stats: {best_ptxas_metrics}\n\n"
                f"Previous Run's Hardware Metrics: {relevant_metrics_dict}"
            )
            if not analysis_response or "DETAILED_PLAN:" not in analysis_response:
                status, details = "Failed (Analysis)", "Analysis Agent did not return a valid plan."
                print(f"❌ {status} {details}")
                continue 
            detailed_plan = analysis_response.split("DETAILED_PLAN:")[1].strip()

            # 4. Coder Agent
            print("[Coder Agent] Generating new kernel...")
            coder_response = agents.call_llm(
                "coder", 
                prompts.CODER_SYSTEM_PROMPT,
                f"Original C++/CUDA Source:\n{best_kernel_code_cuda}\n\nDetailed Plan:\n{detailed_plan}"
            )
            new_kernel_code = extract_code(coder_response)
            if not new_kernel_code:
                status, details = "Failed (Coder)", "Coder Agent did not produce valid code."
                print(f"❌ {status} {details}")
                continue 
            print("[Coder Agent] New kernel source generated.")
                
            # 5. 验证和分析
            current_module_name = f"gemm_evolved_{i}" 
            
            # [!!! 修复 1 !!!] 使用 current_module_name
            print(f"Compiling new kernel (module: {current_module_name})...")
            
            try:
                module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
                    cpp_source, 
                    new_kernel_code, 
                    module_name=current_module_name
                )
                print("Compilation successful.")
                new_ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
                
                is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
                if not is_correct:
                    status, details = "Failed (Correctness)", "New kernel is INCORRECT."
                    print(f"❌ {status}")
                    continue 
                    
            except Exception as e:
                status, details = "Failed (Compilation)", str(e)
                print(f"❌ {status}")
                continue 
                
            print("New kernel is CORRECT. Benchmarking...")
            
            new_time_ms = cuda_utils.benchmark_kernel(A_torch, B_torch)
            print("Analyzing new kernel with NCU...")
            new_ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, 
                current_module_name, 
                N
            )
            
            if new_time_ms < best_time_ms:
                status = "Success (New Best)"
                details = f"Performance improved from {best_time_ms:.3f} ms to {new_time_ms:.3f} ms."
                print(f"✅ {status} {details}")
                
                best_time_ms = new_time_ms
                best_kernel_code_cuda = new_kernel_code
                best_ptxas_metrics = new_ptxas_metrics
                best_ncu_metrics = new_ncu_metrics
            else:
                status = "Failed (Performance Regression)"
                details = f"New time {new_time_ms:.3f} ms is not better than best time {best_time_ms:.3f} ms."
                print(f"❌ {status} {details}")
            
            current_ncu_metrics = new_ncu_metrics

        except Exception as e:
            status, details = "Failed (Unhandled Exception)", str(e)
            print(f"❌ {status} {details}")
            
        finally:
            history_entry = {
                "round": i,
                "goal": opt_goal,
                "status": status,
                "time_ms": new_time_ms if new_time_ms != float('inf') else None,
                "ptxas_metrics": new_ptxas_metrics,
                "details": details
            }
            optimization_history.append(history_entry)

    # 4. 最终报告
    print("\n--- Optimization Finished ---")
    print(f"Baseline performance (Round 0): {optimization_history[0]['time_ms']:.3f} ms")
    print(f"Best kernel performance: {best_time_ms:.3f} ms")
    
    final_kernel_path = "best_gemm_kernel.cu"
    with open(final_kernel_path, "w") as f:
        f.write(best_kernel_code_cuda)
    print(f"Best kernel C++/CUDA source saved to {final_kernel_path}")
    
    with open(config.HISTORY_FILE, 'w') as f:
        json.dump(optimization_history, f, indent=2)
    print(f"Optimization history saved to {config.HISTORY_FILE}")
    
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