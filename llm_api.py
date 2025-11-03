import openai
import config
import kernels # <--- [已更新] 导入 kernels

# 全局初始化一次客户端 (仅在非模拟模式下)
client_dmxapi = None
if not config.MOCK_LLM_CALLS:
    if config.DMX_API_KEY == "YOUR_DMX_API_KEY_HERE" or not config.DMX_API_KEY:
        raise ValueError("Please set your DMX_API_KEY in config.py")
    
    client_dmxapi = openai.OpenAI(
        api_key=config.DMX_API_KEY,
        base_url=config.DMX_API_BASE_URL
    )

def call_llm(system_prompt, user_prompt):
    """
    A simple wrapper for the LLM API call.
    Includes mock logic for testing.
    (已更新为使用 dmxapi 并兼容 OpenAI 格式)
    """
    if config.MOCK_LLM_CALLS:
        print("--- [MOCK LLM CALL] ---")
        
        # Planner 和 Tool 的模拟回复 (不变)
        if "Planner Agent" in system_prompt:
            return "OPTIMIZATION_GOAL: Implement 16x16 tiling using shared memory."
        if "Tool Agent" in system_prompt:
            return "METRICS: ['dram__bytes_read.sum', 'l1tex__t_bytes_read.sum', 'shared_load_transactions.sum', 'achieved_occupancy.avg', 'warp_execution_efficiency.pct']"
        if "Analysis Agent" in system_prompt:
            return """
            DETAILED_PLAN:
            1. Define a preprocessor macro `BLOCK_SIZE` with value 16 (if not already defined).
            2. Declare two shared memory arrays: `__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];` and `__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];`.
            3. Initialize an accumulator register `float C_value = 0.0f;`.
            4. Calculate the number of tiles: `int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;`.
            5. Start a loop that iterates over the tiles (`for (int t = 0; t < num_tiles; ++t)`).
            6. Inside the loop, load data from global A and B into As and Bs, handling boundary conditions (if `A_row < N && A_col < N`).
            7. Call `__syncthreads();` after loading both tiles.
            8. Perform the matrix multiplication for the tile, accumulating the result in `C_value`.
            9. Call `__syncthreads();` after computation, before the next tile load.
            10. After the tile loop, write the final `C_value` to global memory `C[row * N + col]`.
            """
        
        # [!!! 已更新 !!!]
        if "Coder Agent" in system_prompt:
            # 返回一个预先编写好的、完整的 C++/CUDA Tiled 源码
            return f"```cuda\n{kernels.TILED_CUDA_SOURCE}\n```"
        
        return "Mocked response."

    # --- 真实的 API 调用 (使用 dmxapi 客户端) ---
    global client_dmxapi
    if not client_dmxapi:
         raise ValueError("DMX API client is not initialized. Check config.")

    try:
        response = client_dmxapi.chat.completions.create(
            model=config.DMX_MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
            top_p=0.1,
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling DMX API: {e}")
        return None