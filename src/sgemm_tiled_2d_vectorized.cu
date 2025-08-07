#include "sgemm.cuh"
#include <cstdio>

// Tiled 2D SGEMM kernel with vectorized accesses to shared memory
__global__ void sgemm_tiled_2d_vectorized(int M, int N, int K, float alpha,
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

    // Shared-memory buffers for A and B tiles (we transpose B)
    __shared__ __align__(sizeof(float) * VEC_SIZE) float As[BM * BK];
    __shared__ __align__(sizeof(float) * VEC_SIZE) float Bs_t[BK * BN];

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
        for (int tn = 0; tn < TN; ++tn) {
            if ((bx * BN + tx * TN + tn < N) && (k + ty < K)) { // bounds check
                // The "untransposed" access would have been:
                // Bs[BN * ty + TN * tx + tn]
                Bs_t[BK * (TN * tx + tn) + ty] =
                    B[B_tile_offs + N * ty + TN * tx + tn];
            } else {
                Bs_t[BK * (TN * tx + tn) + ty] = 0.0f; // out of bounds
            }
        }
        B_tile_offs += N * BK;
        __syncthreads();

        DTypeVector *Bs_t_vec = reinterpret_cast<DTypeVector *>(Bs_t);
        DTypeVector *As_vec = reinterpret_cast<DTypeVector *>(As);

        // Each thread computes a TMxTN block
        float tmp[TM][TN] = {0.0f};

        /* This loop read elements from As_vec and Bs_t_vec with vector load
        operations. Vector operations, e.g., ld.shared.v4.f32, can be check
        with: 'cuobjdump --dump-ptx <binary_file>'
        However, the performance seems to be lower than the non-vectorized
        version. I assume this is due to the transposed vector Bs_t:
        The access to Bs_t_vec causes bank conflicts. This can be seen with:
        ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared
        --kernel-name sgemm_tiled_2d <binary_file> */
        for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
                for (int bk = 0; bk < BK; bk += VEC_SIZE) {
                    // Load VEC_SIZE elements from As and Bs into resgisters
                    DTypeVector b_vec =
                        Bs_t_vec[(BK * (TN * tx + tn) + bk) / VEC_SIZE];
                    DTypeVector a_vec =
                        As_vec[(BK * (TM * ty + tm) + bk) / VEC_SIZE];

                    const float *a_data =
                        reinterpret_cast<const float *>(&a_vec);
                    const float *b_data =
                        reinterpret_cast<const float *>(&b_vec);
                    
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        tmp[tm][tn] += a_data[i] * b_data[i];
                    }
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
