#include "softmax/softmax.cuh"

// row-wise softmax
__global__ void softmax_simple(int M, int N, const float *A, float *C) {
    // Position in array C from a global perspective:

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    float sum = 0.0f;

    for (int col = 0; col < N; col++) {
        sum += expf(A[x * N + col]);
    }
    C[x * N + y] = expf(A[x * N + y]) / sum;
}