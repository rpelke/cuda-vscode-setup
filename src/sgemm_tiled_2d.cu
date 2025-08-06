#include "sgemm.cuh"
#include <cstdio>

// Tiled 2D SGEMM kernel: C = alpha * A @ B + beta * C
__global__ void sgemm_tiled_2d(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    // Pointers to the block's top-left (position in C)
    const int C_tile_offs = N * BM * by + BN * bx;

    // Offset to row=0 and col=bx in B
    int B_tile_offs = BN * bx;
    // Offset to row=by and col=0 in A
    int A_tile_offs = BM * K * by;

    // C += beta * C
    // This could also be done later but I use C to store intermediate results
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            C[C_tile_offs + N * (TM * ty + tm) + TN * tx + tn] *= beta;
        }
    }

    // Shared-memory buffers for A and B tiles
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // k = {0, BK, 2*BK, ...}
    for (int k = 0; k < K; k += BK) {
        // Each thread loads TM values into As
        for (int tm = 0; tm < TM; ++tm) {
            As[BK * (TM * ty + tm) + tx] =
                A[A_tile_offs + K * (TM * ty + tm) + tx];
        }
        A_tile_offs += BK;

        // Each thread loads TN values into Bs
        for (int tn = 0; tn < TN; ++tn) {
            Bs[BN * ty + TN * tx + tn] = B[B_tile_offs + N * ty + TN * tx + tn];
        }
        B_tile_offs += N * BK;
        __syncthreads();

        // Each thread computes a TMxTN block
        float tmp[TM][TN] = {0.0f};
        for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
                for (int bk = 0; bk < BK; ++bk) {
                    tmp[tm][tn] += As[BK * (TM * ty + tm) + bk] *
                                   Bs[TN * tx + tn + BN * bk];
                }
            }
        }

        // Each thread copies its part of the block to C
        for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
                C[C_tile_offs + N * (TM * ty + tm) + TN * tx + tn] =
                    alpha * tmp[tm][tn] +
                    C[C_tile_offs + N * (TM * ty + tm) + TN * tx + tn];
            }
        }
        __syncthreads();
    }
}
