#ifndef SGEMM_KERNELS_CUH
#define SGEMM_KERNELS_CUH

#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "sgemm_kernel_dimension.cuh"

constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }
constexpr int VEC_SIZE_04 = sizeof(DTypeVector_04) / sizeof(float);
constexpr int VEC_SIZE_05 = sizeof(DTypeVector_05) / sizeof(float);

const std::string GREEN = "\033[32m";
const std::string RESET = "\033[0m";
const std::string RED = "\033[31m";

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
