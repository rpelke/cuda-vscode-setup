#include "sgemm/sgemm.cuh"
#include <cstdio>

// Tiled 2D SGEMM kernel with vectorized accesses to shared memory
__global__ void sgemm_tiled_2d_vectorized_1(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    const int C_tile_offs = N * BM_04 * by + BN_04 * bx;

    // Offset to row=0 and col=bx in B
    int B_tile_offs = BN_04 * bx;
    // Offset to row=by and col=0 in A
    int A_tile_offs = BM_04 * K * by;

    // C += beta * C
    // This  could also be done later but I use C to store intermediate results
    for (int tm = 0; tm < TM_04; ++tm) {
        for (int tn = 0; tn < TN_04; ++tn) {
            if ((bx * BN_04 + tx * TN_04 + tn < N) &&
                (by * BM_04 + ty * TM_04 + tm < M)) { // bounds check
                C[C_tile_offs + N * (TM_04 * ty + tm) + TN_04 * tx + tn] *=
                    beta;
            }
        }
    }

    // Shared-memory buffers for A and B tiles (we transpose B)
    __shared__ __align__(sizeof(float) * VEC_SIZE_04) float As[BM_04 * BK_04];
    __shared__ __align__(sizeof(float) * VEC_SIZE_04) float Bs_t[BK_04 * BN_04];

    // k = {0, BK_04, 2*BK_04, ...}
    for (int k = 0; k < K; k += BK_04) {
        // Each thread loads TM_04 values into As
        for (int tm = 0; tm < TM_04; ++tm) {
            if ((k + tx < K) &&
                (by * BM_04 + ty * TM_04 + tm < M)) { // bounds check
                As[BK_04 * (TM_04 * ty + tm) + tx] =
                    A[A_tile_offs + K * (TM_04 * ty + tm) + tx];
            } else {
                As[BK_04 * (TM_04 * ty + tm) + tx] = 0.0f; // out of bounds
            }
        }
        A_tile_offs += BK_04;

        // Each thread loads TN_04 values into Bs_t
        for (int tn = 0; tn < TN_04; ++tn) {
            if ((bx * BN_04 + tx * TN_04 + tn < N) &&
                (k + ty < K)) { // bounds check
                // The "untransposed" access would have been:
                // Bs[BN_04 * ty + TN_04 * tx + tn]
                Bs_t[BK_04 * (TN_04 * tx + tn) + ty] =
                    B[B_tile_offs + N * ty + TN_04 * tx + tn];
            } else {
                Bs_t[BK_04 * (TN_04 * tx + tn) + ty] = 0.0f; // out of bounds
            }
        }
        B_tile_offs += N * BK_04;
        __syncthreads();

        DTypeVector_04 *Bs_t_vec = reinterpret_cast<DTypeVector_04 *>(Bs_t);
        DTypeVector_04 *As_vec = reinterpret_cast<DTypeVector_04 *>(As);

        // Each thread computes a TM_04xTN_04 block
        float tmp[TM_04][TN_04] = {0.0f};

        /* This loop read elements from As_vec and Bs_t_vec with vector load
        operations. Vector operations, e.g., ld.shared.v4.f32, can be check
        with: 'cuobjdump --dump-ptx <binary_file>'
        However, the performance seems to be lower than the non-vectorized
        version. I assume this is due to the transposed vector Bs_t:
        The access to Bs_t_vec causes bank conflicts. This can be seen with:
        ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared
        --kernel-name sgemm_tiled_2d <binary_file> */
        for (int tm = 0; tm < TM_04; ++tm) {
            for (int tn = 0; tn < TN_04; ++tn) {
                for (int bk = 0; bk < BK_04; bk += VEC_SIZE_04) {
                    // Load VEC_SIZE_04 elements from As and Bs into resgisters
                    DTypeVector_04 b_vec =
                        Bs_t_vec[(BK_04 * (TN_04 * tx + tn) + bk) /
                                 VEC_SIZE_04];
                    DTypeVector_04 a_vec =
                        As_vec[(BK_04 * (TM_04 * ty + tm) + bk) / VEC_SIZE_04];

                    const float *a_data =
                        reinterpret_cast<const float *>(&a_vec);
                    const float *b_data =
                        reinterpret_cast<const float *>(&b_vec);

#pragma unroll
                    for (int i = 0; i < VEC_SIZE_04; ++i) {
                        tmp[tm][tn] += a_data[i] * b_data[i];
                    }
                }
            }
        }

        // Each thread copies its part of the block to C
        for (int tm = 0; tm < TM_04; ++tm) {
            for (int tn = 0; tn < TN_04; ++tn) {
                if ((bx * BN_04 + tx * TN_04 + tn < N) &&
                    (by * BM_04 + ty * TM_04 + tm < M)) { // bounds check
                    C[C_tile_offs + N * (TM_04 * ty + tm) + TN_04 * tx + tn] =
                        alpha * tmp[tm][tn] +
                        C[C_tile_offs + N * (TM_04 * ty + tm) + TN_04 * tx +
                          tn];
                }
            }
        }
        __syncthreads();
    }
}
