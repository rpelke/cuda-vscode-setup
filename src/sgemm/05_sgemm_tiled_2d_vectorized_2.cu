#include "sgemm/sgemm.cuh"
#include <cstdio>

// Tiled 2D SGEMM kernel with vectorized accesses to shared memory
__global__ void sgemm_tiled_2d_vectorized_2(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    // Pointers to the block's top-left (position in C)
    const int C_tile_offs = N * BM_05 * by + BN_05 * bx;

    // Offset to row=0 and col=bx in B
    int B_tile_offs = BN_05 * bx;
    // Offset to row=by and col=0 in A
    int A_tile_offs = BM_05 * K * by;

    // C += beta * C
    // This could also be done later but I use C to store intermediate results
    for (int tm = 0; tm < TM_05; ++tm) {
        for (int tn = 0; tn < TN_05; ++tn) {
            if ((bx * BN_05 + tx * TN_05 + tn < N) &&
                (by * BM_05 + ty * TM_05 + tm < M)) { // bounds check
                C[C_tile_offs + N * (TM_05 * ty + tm) + TN_05 * tx + tn] *=
                    beta;
            }
        }
    }

    // Shared-memory buffers for A and B tiles
    __shared__ __align__(sizeof(float) * VEC_SIZE_05) float As[BM_05 * BK_05];
    __shared__ __align__(sizeof(float) * VEC_SIZE_05) float Bs[BK_05 * BN_05];

    DTypeVector_05 *Bs_vec = reinterpret_cast<DTypeVector_05 *>(Bs);
    const DTypeVector_05 *B_vec = reinterpret_cast<const DTypeVector_05 *>(B);

    // k = {0, BK_05, 2*BK_05, ...}
    for (int k = 0; k < K; k += BK_05) {
        // Each thread loads TM_05 values into As
        for (int tm = 0; tm < TM_05; ++tm) {
            if ((k + tx < K) &&
                (by * BM_05 + ty * TM_05 + tm < M)) { // bounds check
                As[BK_05 * (TM_05 * ty + tm) + tx] =
                    A[A_tile_offs + K * (TM_05 * ty + tm) + tx];
            } else {
                As[BK_05 * (TM_05 * ty + tm) + tx] = 0.0f; // out of bounds
            }
        }
        A_tile_offs += BK_05;

        // Each thread loads TN_05 values into Bs
        int Bs_offs = BN_05 * ty + TN_05 * tx;
        int B_offs = B_tile_offs + N * ty + TN_05 * tx;
        int check_bounds_offs = bx * BN_05 + tx * TN_05;

        for (int tn = 0; tn < TN_05; tn += VEC_SIZE_05) {
            if ((check_bounds_offs + tn + VEC_SIZE_05 - 1 < N) &&
                (k + ty < K)) { // bounds check

                int B_ptr_offs = (B_offs + tn) % VEC_SIZE_05;
                if (B_ptr_offs == 0) { // vector load-store
                    Bs_vec[(Bs_offs + tn) / VEC_SIZE_05] =
                        B_vec[(B_offs + tn) / VEC_SIZE_05];
                } else { // unaligned access
                    for (int i = 0; i < VEC_SIZE_05; ++i) {
                        Bs[Bs_offs + tn + i] = B[B_offs + tn + i];
                    }
                }
            } else {
                for (int i = 0; i < VEC_SIZE_05; ++i) {
                    if ((check_bounds_offs + tn + i < N) && (k + ty) < K) {
                        Bs[Bs_offs + tn + i] = B[B_offs + tn + i];
                    } else {
                        Bs[Bs_offs + tn + i] = 0.0f;
                    }
                }
            }
        }

        B_tile_offs += N * BK_05;
        __syncthreads();

        // Each thread computes a TM_05xTN_05 block
        float tmp[TM_05][TN_05] = {0.0f};
        for (int tm = 0; tm < TM_05; ++tm) {
            for (int tn = 0; tn < TN_05; ++tn) {
                for (int bk = 0; bk < BK_05; ++bk) {
                    tmp[tm][tn] += As[BK_05 * (TM_05 * ty + tm) + bk] *
                                   Bs[TN_05 * tx + tn + BN_05 * bk];
                }
            }
        }

        // Each thread copies its part of the block to C
        for (int tm = 0; tm < TM_05; ++tm) {
            for (int tn = 0; tn < TN_05; ++tn) {
                if ((bx * BN_05 + tx * TN_05 + tn < N) &&
                    (by * BM_05 + ty * TM_05 + tm < M)) { // bounds check
                    C[C_tile_offs + N * (TM_05 * ty + tm) + TN_05 * tx + tn] =
                        alpha * tmp[tm][tn] +
                        C[C_tile_offs + N * (TM_05 * ty + tm) + TN_05 * tx +
                          tn];
                }
            }
        }
        __syncthreads();
    }
}
