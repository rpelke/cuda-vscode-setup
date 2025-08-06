#include "sgemm.cuh"

// Tiled SGEMM kernel: C = alpha * A @ B + beta * C
__global__ void sgemm_tiled(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // Block's position inside C
    int b_y = blockIdx.y;
    int b_x = blockIdx.x;

    // Thread's position inside the block (tile)
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;

    // Pointers to the block's top-left (position in C)
    const int C_tile_offs = (BLOCKSIZE * N * b_y) + (BLOCKSIZE * b_x);
    // Offset to row=0 and col=b_x in B
    const int B_tile_offs = BLOCKSIZE * b_x;
    // Offset to row=b_y and col=0 in A
    const int A_tile_offs = (BLOCKSIZE * K) * b_y;

    // Shared-memory buffers for A and B tiles
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    float tmp = 0.0f;

    // k={0,31,63,...}
    for (int k = 0; k < K; k += BLOCKSIZE) {
        // Load one A value and one B value per thread to shared memory
        if ((k + t_x < K) && (b_y * BLOCKSIZE + t_y < M)) {
            As[BLOCKSIZE * t_y + t_x] = A[A_tile_offs + K * t_y + t_x];
        } else {
            As[BLOCKSIZE * t_y + t_x] = 0.0f; // Out of bounds
        }
        A += BLOCKSIZE;

        if ((b_x * BLOCKSIZE + t_x < N) && (k + t_y < K)) {
            Bs[BLOCKSIZE * t_y + t_x] = B[B_tile_offs + N * t_y + t_x];
        } else {
            Bs[BLOCKSIZE * t_y + t_x] = 0.0f; // Out of bounds
        }
        B += BLOCKSIZE * N;

        // Block threads to wait for all threads to load their values
        __syncthreads();

        // Execute dot product for the current cached block
        for (int i = 0; i < BLOCKSIZE; ++i) {
            tmp += As[BLOCKSIZE * t_y + i] * Bs[BLOCKSIZE * i + t_x];
        }

        __syncthreads();
    }

    if ((b_x * BLOCKSIZE + t_x < N) && (b_y * BLOCKSIZE + t_y < M)) {
        C[C_tile_offs + N * t_y + t_x] =
            alpha * tmp + beta * C[C_tile_offs + N * t_y + t_x];
    }
}
