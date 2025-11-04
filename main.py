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

# [!!! 已删除 !!!]
# def get_diverse_champions(history: list, current_best_code: str, num_kernels=2) -> str:
#     ... (此功能被 summarize_tree_context 替代)

# [!!! 已删除 !!!]
# def summarize_history(history: list) -> str:
#     ... (此功能被 summarize_tree_context 替代)


# [!!! 已更新 !!!] 解决了 TODO 问题 5 和 6（根据您的最新要求）
def format_selected_ncu_metrics(entry):
    """
    一个辅助函数，用于格式化所选的 NCU 指标以包含在摘要中。
    """
    selected_metrics = entry.get('selected_ncu_metrics')
    all_ncu = entry.get('all_ncu_metrics')
    
    if isinstance(selected_metrics, list) and isinstance(all_ncu, dict) and selected_metrics:
        metric_summary = "    Selected NCU Metrics (for Goal):\n"
        for metric_name in selected_metrics:
            value = all_ncu.get(metric_name, 'N/A')
            metric_summary += f"      - {metric_name}: {value}\n"
        return metric_summary
    return ""

# [!!! 已更新 !!!] 解决了 TODO 问题 6
def summarize_tree_context(history: list, best_node: dict, max_ancestors=5, max_children=10) -> str:
    """
    基于当前的最佳节点，生成用于提示的"树上下文"。
    包括"近期成功路径"（祖先）和"近期失败尝试"（子节点）。
    
    [!!! 已更新 !!!]
    - 成功路径现在包含 'Selected NCU Metrics'。
    - 失败尝试现在包含 'Selected NCU Metrics'。
    - 失败尝试在 'Failed (Compilation)' 或 'Failed (Correctness)' 时
      会 *智能地包含* 'Failed Code:'。
    """
    if not best_node:
        return "No optimization history (starting from baseline)."

    # 1. 创建一个 map 以便快速查找
    history_map = {entry['round']: entry for entry in history}
    
    # 2. 生成 "Recent Success Path" (祖先)
    success_path = []
    current_node = best_node
    parent_round = current_node.get('parent_round', -1)
    
    while parent_round != -1 and len(success_path) < max_ancestors:
        if parent_round not in history_map:
            break # 找到了孤儿节点，停止
        current_node = history_map[parent_round]
        
        entry_summary = (
            f"  (Round {current_node['round']}, Time: {current_node.get('time_ms', 0):.3f} ms)\n"
            f"    Goal: {current_node['goal']}\n"
        )
        # [!!! 新增 !!!] 添加祖先节点的选定 NCU 指标
        entry_summary += format_selected_ncu_metrics(current_node)
        
        success_path.append(entry_summary)
        parent_round = current_node.get('parent_round', -1)
        
    success_path.reverse() # 从 Root -> Best
    
    summary_str = "--- Recent Success Path (Root -> Current Best) ---\n"
    if not success_path:
        summary_str += "  (Current Best is Baseline)\n"
    else:
        summary_str += "\n".join(success_path)
        summary_str += f"\n  (Round {best_node['round']}, Current Best, Time: {best_node.get('time_ms', 0):.3f} ms)\n"
        # [!!! 新增 !!!] 添加最佳节点*本身*的选定 NCU 指标
        summary_str += format_selected_ncu_metrics(best_node)

    
    # 3. 生成 "Recent Failed Attempts" (子节点)
    failed_children = []
    best_round_id = best_node['round']
    
    # 反向迭代历史记录以首先获取最近的失败
    for entry in reversed(history):
        if entry.get('parent_round') == best_round_id and "Success" not in entry['status']:
            entry_summary = (
                f"  (Round {entry['round']})\n"
                f"    Goal: {entry['goal']}\n"
                f"    Status: {entry['status']}\n"
                f"    Details: {entry['details']}\n"
            )
            
            # [!!! 新增 !!!] 添加失败尝试的选定 NCU 指标
            entry_summary += format_selected_ncu_metrics(entry)
            
            # [!!! 新增 !!!] 智能代码包含
            if "Compilation" in entry['status'] or "Correctness" in entry['status']:
                failed_code = entry.get('code', '// Code not saved.')
                if failed_code:
                     entry_summary += f"    Failed Code:\n{failed_code}\n"
            
            failed_children.append(entry_summary)
            if len(failed_children) >= max_children:
                break
    
    failed_children.reverse() # 重新按时间顺序
    
    summary_str += "\n\n--- Recent Failed Attempts (Based on this Best Kernel) ---\n"
    if not failed_children:
        summary_str += "  (No failed attempts recorded for this kernel yet.)\n"
    else:
        summary_str += "\n".join(failed_children)

    return summary_str


# [!!! 已更新 !!!] 解决了 TODO 问题 7 (来自上一个请求)
def format_metrics_for_llm(ptxas_metrics: dict, ncu_metrics: dict) -> str:
    """
    [!!! 已更新 !!!] 解决了 TODO 问题 7。
    此函数现在动态地将 *所有* 捕获的 NCU 指标传递给 Planner Agent，
    而不是硬编码一个固定的 "Key" 列表。
    """
    if not ncu_metrics:
        return "Hardware metrics are not yet available."
        
    summary = "=== PTXAS Compiler Metrics ===\n"
    summary += json.dumps(ptxas_metrics, indent=2)
    
    # [!!! 更改 !!!] 直接使用完整的 ncu_metrics 字典，并将标题更改为 "Full Set"
    summary += "\n\n=== NCU Hardware Metrics (Full Set) ===\n" 
    summary += json.dumps(ncu_metrics, indent=2)
    
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
    
    # [!!! 已更新 !!!] 切换到基于节点(Node)的跟踪
    best_node = None
    current_ncu_metrics = {} # 保持不变：用于 Tool Agent
    
    optimization_history = []
    
    if os.path.exists(config.HISTORY_FILE):
        print(f"Loading existing history from {config.HISTORY_FILE}")
        with open(config.HISTORY_FILE, 'r') as f:
            optimization_history = json.load(f)
        
        found_best = False
        # [!!! 已更新 !!!] 查找性能最佳的节点
        best_time_so_far = float('inf')
        for entry in optimization_history:
             if ("Success" in entry['status']) and entry.get('code'):
                entry_time = entry.get('time_ms', float('inf'))
                if entry_time < best_time_so_far:
                    best_time_so_far = entry_time
                    best_node = entry # <--- 找到最佳节点
                    found_best = True
        
        if found_best:
            print(f"Restored best kernel from history (Round {best_node['round']}, Time: {best_node['time_ms']:.3f} ms)")
            # [!!! 已更新 !!!] 恢复上一轮的 NCU 指标以供 Tool Agent 使用
            current_ncu_metrics = best_node.get('all_ncu_metrics', {})
        else:
             print("No successful kernel found in history, starting from baseline.")
             optimization_history = [] 
             
    # 2. 获取基线性能 (Round 0)
    if not optimization_history: 
        print("\n--- Round 0: Compiling and analyzing baseline (naive) kernel ---")
        current_module_name = "gemm_evolved_0"
        baseline_code = kernels.NAIVE_CUDA_SOURCE
        
        try:
            module, stdout_log, stderr_log = cuda_utils.load_gemm_module(
                cpp_source, 
                baseline_code, 
                module_name=current_module_name
            )
            print("Baseline kernel compiled successfully.")
            ptxas_metrics = cuda_utils.parse_ptxas_info(stdout_log + stderr_log)
            
            is_correct = cuda_utils.check_correctness(A_torch, B_torch, C_ref_torch)
            if not is_correct:
                print("❌ Baseline kernel is INCORRECT. Exiting.")
                return
                
            print("Baseline kernel is correct. Benchmarking...")
            time_ms = cuda_utils.benchmark_kernel(A_torch, B_torch)
            
            print("Analyzing baseline kernel with NCU (this may take a while)...")
            ncu_metrics = cuda_utils.get_real_ncu_metrics(
                module.__file__, current_module_name, N
            )
            current_ncu_metrics = ncu_metrics # <--- 设置 "上一轮" 指标
            
            # [!!! 已更新 !!!] 解决了 TODO 问题 5 和 6
            history_entry = {
                "round": 0, 
                "parent_round": -1, # <--- 树的根节点
                "goal": "Baseline", 
                "status": "Success",
                "time_ms": time_ms, 
                "ptxas_metrics": ptxas_metrics,
                "all_ncu_metrics": ncu_metrics,
                "selected_ncu_metrics": [], # <--- 基线没有选择指标
                "details": "Initial baseline measurement",
                "code": baseline_code 
            }
            optimization_history.append(history_entry)
            best_node = history_entry # <--- 基线是当前的最佳节点
            print(f"Baseline performance: {time_ms:.3f} ms")

        except Exception as e:
            print(f"❌ Baseline kernel failed compilation or runtime. Exiting. \n{e}")
            return
    
    # 确保我们有 "best_node"
    if not best_node:
        print("❌ 错误：未能初始化 best_node。历史记录可能已损坏。")
        return
        
    # 确保我们有 "current_ncu_metrics"
    if not current_ncu_metrics: 
        current_ncu_metrics = best_node.get('all_ncu_metrics', {})


    # 3. 开始优化循环
    for i in tqdm(range(len(optimization_history), config.ITERATION_ROUNDS + 1), desc="Optimization Rounds"):
        if i == 0: continue # Round 0 已经完成
        
        print(f"\n--- Round {i}/{config.ITERATION_ROUNDS} ---")
        
        # [!!! 已更新 !!!] 
        # 1. 确定此轮的父节点
        parent_node = best_node
        parent_round_id = parent_node['round']
        parent_kernel_code = parent_node['code']
        parent_time_ms = parent_node['time_ms']

        # 2. 生成新的树上下文（现在包含指标和智能代码）
        history_summary = summarize_tree_context(optimization_history, parent_node)
        
        # 3. 格式化父节点的指标
        metrics_summary = format_metrics_for_llm(parent_node['ptxas_metrics'], parent_node['all_ncu_metrics'])
        
        print("------------------LXT:metrics_summary (to Planner)----------------------")
        print(metrics_summary)
        print("------------------LXT:metrics_summary (to Planner)----------------------")
        
        opt_goal = "N/A"
        bottleneck_analysis = "N/A" 
        detailed_plan = "N/A"
        new_kernel_code = None
        status = "Failed (Unknown)"
        details = ""
        new_time_ms = float('inf')
        new_ptxas_metrics = {}
        new_ncu_metrics = {}
        relevant_metric_names = [] 
        
        try:
            # 1. Planner Agent
            print("[Planner Agent] Analyzing hardware metrics and history...")
            planner_response = agents.call_llm(
                "planner", 
                prompts.PLANNER_SYSTEM_PROMPT,
                # [!!! 已更新 !!!] 使用新的树上下文
                f"Optimization Tree Context:\n{history_summary}\n\n"
                f"=== Hardware Metrics for Current Best Kernel (Round {parent_round_id}) ===\n{metrics_summary}\n\n"
                f"Current Best C++/CUDA Source (Time: {parent_time_ms:.3f} ms):\n{parent_kernel_code}"
            )
            if not planner_response or "OPTIMIZATION_GOAL:" not in planner_response:
                status, details = "Failed (Planner)", "Planner did not return a valid goal."
                print(f"❌ {status} {details}")
                continue 
            
            if "BOTTLENECK_ANALYSIS:" in planner_response:
                 bottleneck_analysis = planner_response.split("BOTTLENECK_ANALYSIS:")[1].split("OPTIMIZATION_GOAL:")[0].strip()
                 print(f"[Planner Agent] Bottleneck identified: {bottleneck_analysis}")
            else:
                status, details = "Failed (Planner)", "Planner did not output BOTTLENECK_ANALYSIS."
                print(f"❌ {status} {details}")
                continue
                 
            opt_goal = planner_response.split("OPTIMIZATION_GOAL:")[1].strip()
            print(f"[Planner Agent] Goal: {opt_goal}")
            print("-----------------------LXT:planner_response----------------------")
            print(planner_response)
            print("-----------------------LXT:planner_response----------------------")
            
            # 2. Tool Agent
            print("[Tool Agent] Selecting metrics...")
            all_metric_names = list(current_ncu_metrics.keys())
            print("-----------------------LXT:all_metric_names----------------------")
            print(all_metric_names)
            print("-----------------------LXT:all_metric_names----------------------")
            if not all_metric_names:
                all_metric_names = config.BASE_NCU_METRICS_LIST_EXAMPLE
                
            tool_response = agents.call_llm(
                "tool", 
                prompts.TOOL_SYSTEM_PROMPT,
                f"All Available NCU Metric Names ({len(all_metric_names)}): {all_metric_names}\n\nOptimization Goal: {opt_goal}"
            )
            print("-----------------------LXT:tool_response----------------------")
            print(tool_response)
            print("-----------------------LXT:tool_response----------------------")
            
            relevant_metric_names = extract_metrics(tool_response)
            
            if not relevant_metric_names:
                status, details = "Failed (Tool)", "Tool Agent did not return a valid metric list."
                print(f"❌ {status} {details}")
                continue 
            print(f"[Tool Agent] Selected {len(relevant_metric_names)} metrics: {relevant_metric_names}")
            
            # [!!! 已更新 !!!] 指标来自父节点
            relevant_metrics_dict = {
                metric: parent_node.get('all_ncu_metrics', {}).get(metric, 0.0) 
                for metric in relevant_metric_names
            }
            
            # 3. Analysis Agent [!!! 已更新 !!!]
            print("[Analysis Agent] Formulating plan...")
            analysis_response = agents.call_llm(
                "analysis", 
                prompts.ANALYSIS_SYSTEM_PROMPT,
                f"Planner's Bottleneck Analysis: {bottleneck_analysis}\n\n" 
                f"Optimization Goal: {opt_goal}\n\n"
                f"Optimization Tree Context:\n{history_summary}\n\n" # <--- 传入新的树上下文
                f"Current Best C++/CUDA Source:\n{parent_kernel_code}\n\n" # <--- 明确传入父节点代码
                f"Current Best Hardware Metrics (Full Set): {metrics_summary}\n\n" 
                f"Tool-Selected Metrics from *Previous* Run (Values): {relevant_metrics_dict}" 
            )
            print("-----------------------LXT:analysis_response----------------------")
            print(analysis_response)
            print("-----------------------LXT:analysis_response----------------------")
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
                f"Original C++/CUDA Source:\n{parent_kernel_code}\n\nDetailed Plan:\n{detailed_plan}" # <--- 基于父节点代码修改
            )
            print("-----------------------LXT:coder_response----------------------")
            print(coder_response)
            print("-----------------------LXT:coder_response----------------------")
            new_kernel_code = extract_code(coder_response)
            if not new_kernel_code:
                status, details = "Failed (Coder)", "Coder Agent did not produce valid code."
                print(f"❌ {status} {details}")
                continue 
            print("[Coder Agent] New kernel source generated.")
                
            # 5. 验证和分析
            current_module_name = f"gemm_evolved_{i}" 
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
            
            # [!!! 已更新 !!!] 与父节点(best_node)比较
            if new_time_ms < parent_time_ms: 
                status = "Success (New Best)"
                details = f"Performance improved from {parent_time_ms:.3f} ms to {new_time_ms:.3f} ms."
                print(f"✅ {status} {details}")
            else:
                status = "Failed (Performance Regression)"
                details = f"New time {new_time_ms:.3f} ms is not better than parent time {parent_time_ms:.3f} ms."
                print(f"❌ {status} {details}")
            
            current_ncu_metrics = new_ncu_metrics

        except Exception as e:
            status, details = "Failed (Unhandled Exception)", str(e)
            print(f"❌ {status} {details}")
            
        finally:
            # [!!! 已更新 !!!] 解决了 TODO 问题 5 和 6
            # 创建新的历史节点
            history_entry = {
                "round": i,
                "parent_round": parent_round_id, # <--- 设置父节点
                "goal": opt_goal,
                "status": status,
                "time_ms": new_time_ms if new_time_ms != float('inf') else None,
                "ptxas_metrics": new_ptxas_metrics,
                "all_ncu_metrics": new_ncu_metrics,
                "selected_ncu_metrics": relevant_metric_names,
                "details": details,
                "code": new_kernel_code if new_kernel_code else "" 
            }
            optimization_history.append(history_entry)

            # [!!! 已更新 !!!] 如果成功，更新 best_node
            # 我们需要比较 new_time_ms 和 best_node['time_ms'] (全局最佳)
            if status == "Success (New Best)" and new_time_ms < best_node['time_ms']:
                print(f"👑 New Global Best! (Round {i}, Time: {new_time_ms:.3f} ms)")
                best_node = history_entry
            # 如果失败，或者只是比父节点好但不是全局最好，
            # best_node 保持不变，下一轮将从*全局最佳*再次尝试
            # (注意：这里的逻辑是 "始终从全局最佳节点分支")
            # (如果想从 "刚刚成功的父节点" 分支，应使用:
            #  if status == "Success (New Best)": best_node = history_entry)
            # 我们将坚持 "始终从全局最佳分支" 的策略。

    # 4. 最终报告
    print("\n--- Optimization Finished ---")
    if optimization_history:
        print(f"Baseline performance (Round 0): {optimization_history[0].get('time_ms', 0.0):.3f} ms")
    print(f"Best kernel performance (Round {best_node['round']}): {best_node['time_ms']:.3f} ms")
    
    final_kernel_path = "best_gemm_kernel.cu"
    with open(final_kernel_path, "w") as f:
        f.write(best_node['code']) # <--- 写入最佳节点的代码
    print(f"Best kernel C++/CUDA source saved to {final_kernel_path}")
    
    with open(config.HISTORY_FILE, 'w') as f:
        json.dump(optimization_history, f, indent=2)
    print(f"Optimization history saved to {config.HISTORY_FILE}")
    
    # 5. 运行最终基准测试
    print("\n--- Running Final Benchmark ---")
    pytorch_time_ms = cuda_utils.get_pytorch_performance(A_torch, B_torch)
    print(f"PyTorch (torch.matmul) performance: {pytorch_time_ms:.3f} ms")
    print(f"Our best LLM-optimized kernel: {best_node['time_ms']:.3f} ms")
    
    speedup = pytorch_time_ms / best_node['time_ms']
    if best_node['time_ms'] < pytorch_time_ms:
        print(f"SUCCESS: Optimized kernel is {speedup:.2f}x faster than PyTorch!")
    else:
        print(f"Result: PyTorch is {1/speedup:.2f}x faster.")

if __name__ == "__main__":
    main()