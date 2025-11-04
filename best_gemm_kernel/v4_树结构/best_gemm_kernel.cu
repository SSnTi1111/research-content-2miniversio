#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// ------------------------------------------------------------------
// KERNEL: gemm_kernel 
// ------------------------------------------------------------------
__global__ void gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    // 使用共享内存进行分块(tiled)的CUDA矩阵乘法 (GEMM) 内核
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    // 线程和块索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 当前块负责计算的C子块的起始行列
    int row_base = by * BLOCK_SIZE;
    int col_base = bx * BLOCK_SIZE;

    // 多累加器用于并行累积部分和
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    // 遍历所有需要的分块
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // 加载A的子块到共享内存
        int a_row = row_base + ty;
        int a_col = t * BLOCK_SIZE + tx;
        if (a_row < N && a_col < N) {
            Asub[ty][tx] = A[a_row * N + a_col];
        } else {
            Asub[ty][tx] = 0.0f;
        }

        // 加载B的子块到共享内存
        int b_row = t * BLOCK_SIZE + ty;
        int b_col = col_base + tx;
        if (b_row < N && b_col < N) {
            Bsub[ty][tx] = B[b_row * N + b_col];
        } else {
            Bsub[ty][tx] = 0.0f;
        }

        // 同步，确保所有线程都完成加载
        __syncthreads();

        // 计算当前tile的部分点积
        // 采用交错(round-robin)的方式将16个元素分配到4个累加器，打破依赖链
        int kBase = 0;
        for (; kBase + 15 < BLOCK_SIZE; kBase += 16) {
            // 第一组: kBase + [0..3]
            sum0 += Asub[ty][kBase + 0] * Bsub[kBase + 0][tx];
            sum1 += Asub[ty][kBase + 1] * Bsub[kBase + 1][tx];
            sum2 += Asub[ty][kBase + 2] * Bsub[kBase + 2][tx];
            sum3 += Asub[ty][kBase + 3] * Bsub[kBase + 3][tx];

            // 第二组: kBase + [4..7]
            sum0 += Asub[ty][kBase + 4] * Bsub[kBase + 4][tx];
            sum1 += Asub[ty][kBase + 5] * Bsub[kBase + 5][tx];
            sum2 += Asub[ty][kBase + 6] * Bsub[kBase + 6][tx];
            sum3 += Asub[ty][kBase + 7] * Bsub[kBase + 7][tx];

            // 第三组: kBase + [8..11]
            sum0 += Asub[ty][kBase + 8] * Bsub[kBase + 8][tx];
            sum1 += Asub[ty][kBase + 9] * Bsub[kBase + 9][tx];
            sum2 += Asub[ty][kBase + 10] * Bsub[kBase + 10][tx];
            sum3 += Asub[ty][kBase + 11] * Bsub[kBase + 11][tx];

            // 第四组: kBase + [12..15]
            sum0 += Asub[ty][kBase + 12] * Bsub[kBase + 12][tx];
            sum1 += Asub[ty][kBase + 13] * Bsub[kBase + 13][tx];
            sum2 += Asub[ty][kBase + 14] * Bsub[kBase + 14][tx];
            sum3 += Asub[ty][kBase + 15] * Bsub[kBase + 15][tx];
        }

        // 处理尾部(当BLOCK_SIZE不是16的倍数时)，进行循环级别的边界控制
        for (int k = kBase; k < BLOCK_SIZE; ++k) {
            // 将尾部累积到其中一个累加器即可
            sum0 += Asub[ty][k] * Bsub[k][tx];
        }

        // 同步，确保所有线程完成计算后再进行下一次加载
        __syncthreads();
    }

    // 合并累加器得到最终结果
    float sum = sum0 + sum1 + sum2 + sum3;

    // 写回结果到全局内存，需越界检查
    int c_row = row_base + ty;
    int c_col = col_base + tx;
    if (c_row < N && c_col < N) {
        C[c_row * N + c_col] = sum;
    }
}

// ------------------------------------------------------------------
// WRAPPER: gemm_cuda (这是PyTorch和CUDA之间的桥梁)
// ------------------------------------------------------------------
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    
    // --- 输入验证 ---
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(M == N && K == N, "This naive example assumes square N=M=K matrices");
    auto C = torch::zeros({M, N}, A.options());

    // --- 内核启动配置 ---
    const int block_dim_x = BLOCK_SIZE;
    const int block_dim_y = BLOCK_SIZE;
    const int grid_dim_x = (N + block_dim_x - 1) / block_dim_x;
    const int grid_dim_y = (N + block_dim_y - 1) / block_dim_y;
    dim3 blocks(grid_dim_x, grid_dim_y);
    dim3 threads(block_dim_x, block_dim_y);

    // --- 启动内核 ---
    gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // --- 错误检查 ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in gemm_kernel: " + std::string(cudaGetErrorString(err)));
    }
    return C;
}