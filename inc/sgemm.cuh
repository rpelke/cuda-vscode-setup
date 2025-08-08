#ifndef SGEMM_KERNELS_CUH
#define SGEMM_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int BK = 16;

using DTypeVector = float4;
constexpr int VEC_SIZE = sizeof(DTypeVector) / sizeof(float);

constexpr int BLOCKSIZE = 32;
constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

void cpu_sgemm(int M, int N, int K, float alpha, const std::vector<float> &A,
               const std::vector<float> &B, float beta, std::vector<float> &C);

__global__ void sgemm_simple(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C);
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C);
__global__ void sgemm_tiled(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);
__global__ void sgemm_tiled_2d(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C);
__global__ void sgemm_tiled_2d_vectorized_1(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C);
__global__ void sgemm_tiled_2d_vectorized_2(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C);

#endif // SGEMM_KERNELS_CUH
