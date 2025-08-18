#ifndef SGEMM_KERNELS_CUH
#define SGEMM_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector>

constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

// Kernel 01
constexpr int BLOCKSIZE = 32;

// Kernel 03
constexpr int BM_03 = 64;
constexpr int BN_03 = 32;
constexpr int TM_03 = 4;
constexpr int TN_03 = 2;
constexpr int BK_03 = 16;

// Kernel 04
constexpr int BM_04 = 64;
constexpr int BN_04 = 32;
constexpr int TM_04 = 4;
constexpr int TN_04 = 2;
constexpr int BK_04 = 16;
using DTypeVector_04 = float2;
constexpr int VEC_SIZE_04 = sizeof(DTypeVector_04) / sizeof(float);

// Kernel 05
constexpr int BM_05 = 64;
constexpr int BN_05 = 32;
constexpr int TM_05 = 4;
constexpr int TN_05 = 2;
constexpr int BK_05 = 16;
using DTypeVector_05 = float2;
constexpr int VEC_SIZE_05 = sizeof(DTypeVector_05) / sizeof(float);

// Kernel 06
constexpr int BM_06 = 64;
constexpr int BN_06 = 32;
constexpr int TM_06 = 4;
constexpr int TN_06 = 2;
constexpr int BK_06 = 16;
constexpr int WN_06 = 16;
constexpr int WM_06 = 16;

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
__global__ void sgemm_warptiling(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C);

#endif // SGEMM_KERNELS_CUH
