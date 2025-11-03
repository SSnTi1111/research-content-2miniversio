import torch
import os # <-- 新增

# 为 PyTorch >= 2.1 设置环境变量 (推荐)
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# --- General Settings ---
MATRIX_N = 2048
ITERATION_ROUNDS = 5  # 可外部设置的迭代轮数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- LLM Settings (dmxapi) ---
DMX_API_KEY = "sk-EurbTIrCe7fqGs8LxhxS77d5wKb0LlmUlPHWUnYa59PTVY9P" # (使用您提供的key)
DMX_MODEL_NAME = "MiniMax-M2" # (使用您提供的模型)
DMX_API_BASE_URL = "https://www.dmxapi.cn/v1"

# --- 模拟开关 ---
# !! 设置为 True 可在不调用LLM API的情况下运行，用于测试流程 !!
MOCK_LLM_CALLS = True 

# --- Profiler Settings ---
# 模拟的27个NCU指标 (保持不变，用于模拟Tool Agent)
ALL_NCU_METRICS_LIST = [
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes_read.sum",
    "lts__t_bytes_write.sum", "l1tex__t_bytes_read.sum", "l1tex__t_bytes_write.sum",
    "sm__cycles_elapsed.avg", "smsp__warps_elapsed.avg", "sm__inst_executed.avg",
    "smsp__thread_inst_executed.avg", "sm__pipe_tensor_inst_executed.avg",
    "sm__inst_executed_pipe_fp16.avg", "sm__inst_executed_pipe_fp32.avg",
    "sm__inst_executed_pipe_fp64.avg", "gld_transactions.sum", "gst_transactions.sum",
    "shared_load_transactions.sum", "shared_store_transactions.sum",
    "l2_read_transactions.sum", "l2_write_transactions.sum",
    "dram_read_transactions.sum", "dram_write_transactions.sum",
    "achieved_occupancy.avg", "achieved_active_warps_per_sm.avg",
    "branch_efficiency.pct", "warp_execution_efficiency.pct", "smsp__sass_inst_executed.sum"
]
assert len(ALL_NCU_METRICS_LIST) == 27