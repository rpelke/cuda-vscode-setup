#include "softmax/softmax.cuh"

// row-wise softmax
__global__ void softmax_block_sum(int M, int N, const float *A, float *C) {
    // Position in array C from a global perspective:
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Shared-memory buffers for A and B tiles
    __shared__ float sums[BLOCKSIZE_00];

    float partialSum = 0.0f;

    // Each thread calculates a partial sum
    for (int col = N / blockDim.y * threadIdx.y; col < min(N / blockDim.y * (threadIdx.y+1), N); col++) {
        partialSum += __expf(A[x * N + col]);
    }

    // Add has to be atomic to avoid race condition
    atomicAdd(&sums[threadIdx.x], partialSum);
    __syncthreads();

    C[x * N + y] = __expf(A[x * N + y]) / sums[threadIdx.x];
}