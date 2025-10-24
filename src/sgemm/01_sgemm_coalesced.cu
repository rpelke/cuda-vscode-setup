#include "sgemm/sgemm.cuh"

// Simple SGEMM with coalesced access: C = alpha * A @ B + beta * C
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
    const unsigned int x =
        (threadIdx.x % BLOCKSIZE_01) + blockIdx.x * BLOCKSIZE_01;
    const unsigned int y =
        (threadIdx.x / BLOCKSIZE_01) + blockIdx.y * BLOCKSIZE_01;

    if (x < N && y < M) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[y * K + k] * B[k * N + x];
        }
        // C = α*(A@B)+β*C
        C[y * N + x] = alpha * tmp + beta * C[y * N + x];
    }
}
