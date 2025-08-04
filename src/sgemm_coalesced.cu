#include "sgemm.cuh"

// Naive SGEMM with coalesced access: C = alpha * A @ B + beta * C
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
    // Position in array C from a global perspective:
    const unsigned int x = (threadIdx.x / BLOCKSIZE) + blockIdx.x * BLOCKSIZE;
    const unsigned int y = (threadIdx.x % BLOCKSIZE) + blockIdx.y * BLOCKSIZE;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[x * K + k] * B[k * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
