#ifndef SGEMM_KERNELS_CUH
#define SGEMM_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector>

void cpu_sgemm(int M, int N, int K, float alpha, const std::vector<float> &A,
               const std::vector<float> &B, float beta, std::vector<float> &C);

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C);
__global__ void sgemm_tiled(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);

constexpr int BLOCKSIZE = 32;
constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

#endif // SGEMM_KERNELS_CUH
