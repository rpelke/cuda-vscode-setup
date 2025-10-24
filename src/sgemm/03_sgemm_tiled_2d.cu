#include "sgemm/sgemm.cuh"
#include <cstdio>

// Tiled 2D SGEMM kernel: C = alpha * A @ B + beta * C
__global__ void sgemm_tiled_2d(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    // Pointers to the block's top-left (position in C)
    const int C_tile_offs = N * BM_03 * by + BN_03 * bx;

    // Offset to row=0 and col=bx in B
    int B_tile_offs = BN_03 * bx;
    // Offset to row=by and col=0 in A
    int A_tile_offs = BM_03 * K * by;

    // C += beta * C
    // This could also be done later but I use C to store intermediate results
    for (int tm = 0; tm < TM_03; ++tm) {
        for (int tn = 0; tn < TN_03; ++tn) {
            if ((bx * BN_03 + tx * TN_03 + tn < N) &&
                (by * BM_03 + ty * TM_03 + tm < M)) { // bounds check
                C[C_tile_offs + N * (TM_03 * ty + tm) + TN_03 * tx + tn] *=
                    beta;
            }
        }
    }

    // Shared-memory buffers for A and B tiles
    __shared__ float As[BM_03 * BK_03];
    __shared__ float Bs[BK_03 * BN_03];

    // k = {0, BK_03, 2*BK_03, ...}
    for (int k = 0; k < K; k += BK_03) {
        // Each thread loads TM_03 values into As
        for (int tm = 0; tm < TM_03; ++tm) {
            if ((k + tx < K) &&
                (by * BM_03 + ty * TM_03 + tm < M)) { // bounds check
                As[BK_03 * (TM_03 * ty + tm) + tx] =
                    A[A_tile_offs + K * (TM_03 * ty + tm) + tx];
            } else {
                As[BK_03 * (TM_03 * ty + tm) + tx] = 0.0f; // out of bounds
            }
        }
        A_tile_offs += BK_03;

        // Each thread loads TN_03 values into Bs
        for (int tn = 0; tn < TN_03; ++tn) {
            if ((bx * BN_03 + tx * TN_03 + tn < N) &&
                (k + ty < K)) { // bounds check
                Bs[BN_03 * ty + TN_03 * tx + tn] =
                    B[B_tile_offs + N * ty + TN_03 * tx + tn];
            } else {
                Bs[BN_03 * ty + TN_03 * tx + tn] = 0.0f; // out of bounds
            }
        }
        B_tile_offs += N * BK_03;
        __syncthreads();

        // Each thread computes a TM_03xTN_03 block
        float tmp[TM_03][TN_03] = {0.0f};
        for (int tm = 0; tm < TM_03; ++tm) {
            for (int tn = 0; tn < TN_03; ++tn) {
                for (int bk = 0; bk < BK_03; ++bk) {
                    tmp[tm][tn] += As[BK_03 * (TM_03 * ty + tm) + bk] *
                                   Bs[TN_03 * tx + tn + BN_03 * bk];
                }
            }
        }

        // Each thread copies its part of the block to C
        for (int tm = 0; tm < TM_03; ++tm) {
            for (int tn = 0; tn < TN_03; ++tn) {
                if ((bx * BN_03 + tx * TN_03 + tn < N) &&
                    (by * BM_03 + ty * TM_03 + tm < M)) { // bounds check
                    C[C_tile_offs + N * (TM_03 * ty + tm) + TN_03 * tx + tn] =
                        alpha * tmp[tm][tn] +
                        C[C_tile_offs + N * (TM_03 * ty + tm) + TN_03 * tx +
                          tn];
                }
            }
        }
        __syncthreads();
    }
}
