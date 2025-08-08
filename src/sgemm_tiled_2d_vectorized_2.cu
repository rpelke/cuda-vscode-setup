#include "sgemm.cuh"
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
    const int C_tile_offs = N * BM * by + BN * bx;

    // Offset to row=0 and col=bx in B
    int B_tile_offs = BN * bx;
    // Offset to row=by and col=0 in A
    int A_tile_offs = BM * K * by;

    // C += beta * C
    // This could also be done later but I use C to store intermediate results
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            if ((bx * BN + tx * TN + tn < N) &&
                (by * BM + ty * TM + tm < M)) { // bounds check
                C[C_tile_offs + N * (TM * ty + tm) + TN * tx + tn] *= beta;
            }
        }
    }

    // Shared-memory buffers for A and B tiles
    __shared__ __align__(sizeof(float) * VEC_SIZE) float As[BM * BK];
    __shared__ __align__(sizeof(float) * VEC_SIZE) float Bs[BK * BN];

    DTypeVector *Bs_vec = reinterpret_cast<DTypeVector *>(Bs);
    const DTypeVector *B_vec = reinterpret_cast<const DTypeVector *>(B);

    // k = {0, BK, 2*BK, ...}
    for (int k = 0; k < K; k += BK) {
        // Each thread loads TM values into As
        for (int tm = 0; tm < TM; ++tm) {
            if ((k + tx < K) && (by * BM + ty * TM + tm < M)) { // bounds check
                As[BK * (TM * ty + tm) + tx] =
                    A[A_tile_offs + K * (TM * ty + tm) + tx];
            } else {
                As[BK * (TM * ty + tm) + tx] = 0.0f; // out of bounds
            }
        }
        A_tile_offs += BK;

        // Each thread loads TN values into Bs
        int Bs_offs = BN * ty + TN * tx;
        int B_offs = B_tile_offs + N * ty + TN * tx;
        int check_bounds_offs = bx * BN + tx * TN;

        for (int tn = 0; tn < TN; tn += VEC_SIZE) {
            if ((check_bounds_offs + tn + VEC_SIZE - 1 < N) &&
                (k + ty < K)) { // bounds check

                int B_ptr_offs = (B_offs + tn) % VEC_SIZE;
                if (B_ptr_offs == 0) { // vector load-store
                    Bs_vec[(Bs_offs + tn) / VEC_SIZE] = B_vec[(B_offs + tn) / VEC_SIZE];
                }
                else { // unaligned access
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        Bs[Bs_offs + tn + i] = B[B_offs + tn + i];
                    }
                }
            } else {
                for (int i = 0; i < VEC_SIZE; ++i) {
                    if ((check_bounds_offs + tn + i < N) && (k + ty) < K) {
                        Bs[Bs_offs + tn + i] = B[B_offs + tn + i];
                    } else {
                        Bs[Bs_offs + tn + i] = 0.0f;
                    }
                }
            }
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
                if ((bx * BN + tx * TN + tn < N) &&
                    (by * BM + ty * TM + tm < M)) { // bounds check
                    C[C_tile_offs + N * (TM * ty + tm) + TN * tx + tn] =
                        alpha * tmp[tm][tn] +
                        C[C_tile_offs + N * (TM * ty + tm) + TN * tx + tn];
                }
            }
        }
        __syncthreads();
    }
}
