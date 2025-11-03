import torch
import os

# 为 PyTorch >= 2.1 设置环境变量 (推荐)
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# --- General Settings ---
MATRIX_N = 2048
ITERATION_ROUNDS = 5  # 可外部设置的迭代轮数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- LLM Settings (dmxapi) ---
DMX_API_KEY = "sk-EurbTIrCe7fqGs8LxhxS77d5wKb0LlmUlPHWUnYa59PTVY9P" # (使用您提供的key)
DMX_MODEL_NAME = "gpt-5-mini" # (您在 config.py 中设置的模型)
DMX_API_BASE_URL = "https://www.dmxapi.cn/v1"

# --- 模拟开关 ---
# !! 设置为 False 以启用真实的 LLM API 调用 !!
MOCK_LLM_CALLS = False

# --- Profiler Settings ---
# (此列表现在仅作为后备，或用于 Tool Agent 的提示词示例)
BASE_NCU_METRICS_LIST_EXAMPLE = [
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes_read.sum",
    "lts__t_bytes_write.sum", "l1tex__t_bytes_read.sum", "l1tex__t_bytes_write.sum",
    "sm__cycles_elapsed.avg", "smsp__warps_elapsed.avg", "sm__inst_executed.avg",
    "smsp__thread_inst_executed.avg", "achieved_occupancy.avg", 
    "branch_efficiency.pct", "warp_execution_efficiency.pct"
]