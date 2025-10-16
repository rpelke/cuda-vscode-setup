#ifndef SOFTMAX_KERNELS_CUH
#define SOFTMAX_KERNELS_CUH

#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "sgemm_kernel_dimension.cuh"

void cpu_softmax(int M, int N, const std::vector<float> &A, std::vector<float> &C);

__global__ void softmax_simple(int M, int N, const float *A, float *C);

__global__ void softmax_block_sum(int M, int N, const float *A, float *C);

__global__ void softmax_block_binary_k0(int M, int N, const float *A, float *C, float *temp);
__global__ void softmax_block_binary_k1(int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0);
__global__ void softmax_block_binary_k2(int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0);

#endif // SOFTMAX_KERNELS_CUH
