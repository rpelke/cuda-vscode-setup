#include "sgemm.cuh"
#include <cstdio>

// Tiled 2D SGEMM kernel with warptiling
// Each thread processes TMxTN block
__global__ void sgemm_warptiling(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    unsigned int wx = threadIdx.x / (WN / TN);
    unsigned int wy = threadIdx.y / (WM / TM);

    unsigned int tx = (threadIdx.x % (WN / TN));
    unsigned int ty = (threadIdx.y % (WM / TM));

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
            C[C_tile_offs + N * (WM * wy + TM * ty + tm) + WN * wx + TN * tx +
              tn] *= beta;
        }
    }

    // Shared-memory buffers for A and B tiles
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // k = {0, BK, 2*BK, ...}
    for (int k = 0; k < K; k += BK) {
        // Each thread loads TM values into As
        for (int tm = 0; tm < TM; ++tm) {
            As[BK * (WM * wy + TM * ty + tm) + (WN / TN) * wx + tx] =
                A[A_tile_offs + K * (WM * wy + TM * ty + tm) + (WN / TN) * wx +
                  tx];
        }
        A_tile_offs += BK;

        // Each thread loads TN values into Bs
        for (int tn = 0; tn < TN; ++tn) {
            Bs[BN * ((WM / TM) * wy + ty) + WN * wx + TN * tx + tn] =
                B[B_tile_offs + N * ((WM / TM) * wy + ty) + WN * wx + TN * tx +
                  tn];
        }
        B_tile_offs += N * BK;
        __syncthreads();

        // Each thread computes a TMxTN block
        float tmp[TM][TN] = {0.0f};
        for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
                for (int bk = 0; bk < BK; ++bk) {
                    tmp[tm][tn] += As[BK * (WM * wy + TM * ty + tm) + bk] *
                                   Bs[BN * bk + WN * wx + TN * tx + tn];
                }
            }
        }

        // Each thread copies its part of the block to C
        for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
                C[C_tile_offs + N * (WM * wy + TM * ty + tm) + WN * wx +
                  TN * tx + tn] = alpha * tmp[tm][tn] +
                                  C[C_tile_offs + N * (WM * wy + TM * ty + tm) +
                                    WN * wx + TN * tx + tn];
            }
        }
        __syncthreads();
    }
}
