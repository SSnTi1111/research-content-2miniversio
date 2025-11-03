#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
static_assert(BLOCK_SIZE * BLOCK_SIZE <= 1024, "BLOCK_SIZE^2 must be <= 1024");

// ------------------------------------------------------------------
// KERNEL: gemm_kernel 
// ------------------------------------------------------------------
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // 2D block-tiled GEMM with shared memory
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    // Cache row*N to avoid repeated multiplications
    const int rowN = row * N;

    // Tile size (equal to BLOCK_SIZE)
    const int TILE = BLOCK_SIZE;

    // Shared-memory tiles
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    // Per-thread accumulator
    float sum = 0.0f;

    // Number of tiles along K dimension
    const int numTiles = (N + TILE - 1) / TILE;

    // Loop over tiles
    for (int t = 0; t < numTiles; ++t) {
        // Global indices to load
        const int A_col = t * TILE + tx; // column in A for this tile
        const int B_row = t * TILE + ty; // row in B for this tile

        // Cooperative loads into shared memory with bounds checks.
        // All threads participate so that __syncthreads is safe.
        Asub[ty][tx] = (row < N && A_col < N) ? A[rowN + A_col] : 0.0f;
        Bsub[ty][tx] = (B_row < N && col < N) ? B[B_row * N + col] : 0.0f;

        // Ensure the entire tile is visible
        __syncthreads();

        // Compute partial products for this tile
        for (int k = 0; k < TILE; ++k) {
            sum += Asub[ty][k] * Bsub[k][tx];
        }

        // Ensure no thread overwrites shared tiles before others finish
        __syncthreads();
    }

    // Write back result with bounds check
    if (row < N && col < N) {
        C[rowN + col] = sum;
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