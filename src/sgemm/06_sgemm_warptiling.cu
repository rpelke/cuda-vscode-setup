#include "sgemm.cuh"
#include <cstdio>
#include <nvtx3/nvToolsExt.h>

// Tiled 2D SGEMM kernel with warptiling
// Each thread processes TM_06xTN_06 block
__global__ void sgemm_warptiling(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    unsigned int wx = threadIdx.x / (WN_06 / TN_06);
    unsigned int wy = threadIdx.y / (WM_06 / TM_06);

    unsigned int tx = (threadIdx.x % (WN_06 / TN_06));
    unsigned int ty = (threadIdx.y % (WM_06 / TM_06));

    // Pointers to the block's top-left (position in C)
    const int C_tile_offs = N * BM_06 * by + BN_06 * bx;
    // Offset to row=0 and col=bx in B
    int B_tile_offs = BN_06 * bx;
    // Offset to row=by and col=0 in A
    int A_tile_offs = BM_06 * K * by;

    // C += beta * C
    // This could also be done later but I use C to store intermediate results
    for (int tm = 0; tm < TM_06; ++tm) {
        for (int tn = 0; tn < TN_06; ++tn) {
            if ((BN_06 * bx + WN_06 * wx + TN_06 * tx + tn < N) &&
                (BM_06 * by + WM_06 * wy + TM_06 * ty + tm < M)) {
                C[C_tile_offs + N * (WM_06 * wy + TM_06 * ty + tm) +
                  WN_06 * wx + TN_06 * tx + tn] *= beta;
            }
        }
    }

    // Shared-memory buffers for A and B tiles
    __shared__ float As[BM_06 * BK_06];
    __shared__ float Bs[BK_06 * BN_06];

    // k = {0, BK_06, 2*BK_06, ...}
    for (int k = 0; k < K; k += BK_06) {
        // Each thread loads TM_06 values into As
        for (int tm = 0; tm < TM_06; ++tm) {
            if ((k + (WN_06 / TN_06) * wx + tx < K) &&
                (BM_06 * by + WM_06 * wy + TM_06 * ty + tm < M)) {
                As[BK_06 * (WM_06 * wy + TM_06 * ty + tm) +
                   (WN_06 / TN_06) * wx + tx] =
                    A[A_tile_offs + K * (WM_06 * wy + TM_06 * ty + tm) +
                      (WN_06 / TN_06) * wx + tx];
            } else {
                As[BK_06 * (WM_06 * wy + TM_06 * ty + tm) +
                   (WN_06 / TN_06) * wx + tx] = 0.0f;
            }
        }
        A_tile_offs += BK_06;

        // Each thread loads TN_06 values into Bs
        for (int tn = 0; tn < TN_06; ++tn) {
            if ((BN_06 * bx + WN_06 * wx + TN_06 * tx + tn < N) &&
                (k + (WM_06 / TM_06) * wy + ty < K)) {
                Bs[BN_06 * ((WM_06 / TM_06) * wy + ty) + WN_06 * wx +
                   TN_06 * tx + tn] =
                    B[B_tile_offs + N * ((WM_06 / TM_06) * wy + ty) +
                      WN_06 * wx + TN_06 * tx + tn];
            } else {
                Bs[BN_06 * ((WM_06 / TM_06) * wy + ty) + WN_06 * wx +
                   TN_06 * tx + tn] = 0.0f;
            }
        }
        B_tile_offs += N * BK_06;
        __syncthreads();

        // Temporary results of the TM_06xTN_06 mini-GEMM within a thread
        float tmp[TM_06][TN_06] = {0.0f};

        // *******************************************************************
        // ***** This part will be discussed in "docs/01_register_blocking.md"
        for (int tm = 0; tm < TM_06; ++tm) {
            for (int tn = 0; tn < TN_06; ++tn) {
                for (int bk = 0; bk < BK_06; ++bk) {
                    tmp[tm][tn] +=
                        As[BK_06 * (WM_06 * wy + TM_06 * ty + tm) + bk] *
                        Bs[BN_06 * bk + WN_06 * wx + TN_06 * tx + tn];
                }
            }
        }
        // *******************************************************************

        // Each thread copies its part of the block to C
        for (int tm = 0; tm < TM_06; ++tm) {
            for (int tn = 0; tn < TN_06; ++tn) {
                if ((bx * BN_06 + WN_06 * wx + TN_06 * tx + tn < N) &&
                    (by * BM_06 + WM_06 * wy + TM_06 * ty + tm < M)) {
                    const unsigned int C_elem_addr =
                        C_tile_offs + N * (WM_06 * wy + TM_06 * ty + tm) +
                        WN_06 * wx + TN_06 * tx + tn;
                    C[C_elem_addr] = alpha * tmp[tm][tn] + C[C_elem_addr];
                }
            }
        }
        __syncthreads();
    }
}

// clang-format off
/* 1-D Blockdimension with (BN_06 / TN_06) * (BM_06 / TM_06) threads.
Launch:
    dim3 blockWarptiling((BN_06 / TN_06) * (BM_06 / TM_06), 1, 1);

Kernel:
    unsigned int wx = threadIdx.x / ((BM_06 / TM_06) * (WN_06 / TN_06));
    unsigned int w_block_num = threadIdx.x / ((WN_06 / TN_06) * (WM_06 / TM_06));
    unsigned int wy = w_block_num % (BM_06 / WM_06);
    unsigned int tx = threadIdx.x % (WN_06 / TN_06);
    unsigned int t_row_id = threadIdx.x / (WN_06 / TN_06);
    unsigned int ty = t_row_id % (WM_06 / TM_06);
*/
// clang-format on
