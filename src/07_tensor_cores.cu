#include "sgemm.cuh"
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// SGEMM using tensor cores for GEMM: C = alpha * A @ B + beta * C
__global__ void sgemm_tensorcores(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    // Pointers to the block's top-left (position in C)
    const int C_tile_offs = N * BLOCKSIZE_07 * by + BLOCKSIZE_07 * bx;

    // Offset to row=0 and col=bx in B
    int B_tile_offs = BLOCKSIZE_07 * bx;
    // Offset to row=by and col=0 in A
    int A_tile_offs = BLOCKSIZE_07 * K * by;

    // Shared-memory buffer for C
    __shared__ float Cs[BLOCKSIZE_07 * BLOCKSIZE_07];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Shared-memory buffers for A and B tiles
    __shared__ half As[BLOCKSIZE_07 * BLOCKSIZE_07];
    __shared__ half Bs[BLOCKSIZE_07 * BLOCKSIZE_07];

    // k = {0, BLOCKSIZE_07, 2*BLOCKSIZE_07, ...}
    for (int k = 0; k < K; k += BLOCKSIZE_07) {
        // Each thread loads TM_07 * TM_07 values into As and Bs
        for (int tm = 0; tm < TM_07; ++tm) {
            for (int tn = 0; tn < TN_07; ++tn) {
                int shmem_offs = BLOCKSIZE_07 * TM_07 * ty + TN_07 * tx;
                if ((k + TN_07 * tx + tn < K) &&
                    (by * BLOCKSIZE_07 + ty * TM_07 + tm < M)) {
                    As[shmem_offs + BLOCKSIZE_07 * tm + tn] =
                        __float2half(A[A_tile_offs + K * (TM_07 * ty + tm) +
                                       TN_07 * tx + tn]);
                } else {
                    As[shmem_offs + BLOCKSIZE_07 * tm + tn] = 0.0;
                }
                if ((k + TM_07 * ty + tm < K) &&
                    (bx * BLOCKSIZE_07 + tx * TN_07 + tn < N)) {
                    Bs[shmem_offs + BLOCKSIZE_07 * tm + tn] =
                        __float2half(B[B_tile_offs + N * (TM_07 * ty + tm) +
                                       TN_07 * tx + tn]);
                } else {
                    Bs[shmem_offs + BLOCKSIZE_07 * tm + tn] = 0.0;
                }
            }
        }
        A_tile_offs += BLOCKSIZE_07;
        B_tile_offs += N * BLOCKSIZE_07;
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
            b_frag;

        wmma::load_matrix_sync(a_frag, As, 16);
        wmma::load_matrix_sync(b_frag, Bs, 16);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(Cs, c_frag, 16, wmma::mem_row_major);

        __syncthreads();
    }

    for (int tm = 0; tm < TM_07; ++tm) {
        for (int tn = 0; tn < TN_07; ++tn) {
            if ((bx * BLOCKSIZE_07 + tx * TN_07 + tn < N) &&
                (by * BLOCKSIZE_07 + ty * TM_07 + tm < M)) {
                int shmem_offs = BLOCKSIZE_07 * TM_07 * ty + TN_07 * tx;
                C[C_tile_offs + N * (TM_07 * ty + tm) + TN_07 * tx + tn] =
                    Cs[shmem_offs + BLOCKSIZE_07 * tm + tn] * alpha +
                    C[C_tile_offs + N * (TM_07 * ty + tm) + TN_07 * tx + tn] *
                        beta;
            }
        }
    }
}
