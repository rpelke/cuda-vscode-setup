#include "softmax/softmax.cuh"

// This impl uses warp shuffle intrinsics for the across-thread reduction
__global__ void softmax_warp_shuffle_k0(int M, int N, const float *A, float *C,
                                        float *temp) {

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= N)
        return;

    int globalStartElem = y * N + x;

    // Full mask for all 32 threads
    unsigned mask = 0xffffffff;

    float val = __expf(A[globalStartElem]);

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (threadIdx.x == 0)
        temp[y * gridDim.x + blockIdx.x] = val;
}

__global__ void softmax_warp_shuffle_k1(int M, int N, const float *A, float *C,
                                        float *temp, int temp_N) {

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= N)
        return;

    int globalStartElem = y * temp_N + x;

    // Full mask for all 32 threads
    unsigned mask = 0xffffffff;

    float val = temp[globalStartElem];

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (threadIdx.x == 0)
        temp[y * temp_N + blockIdx.x] = val;
}

// Use k2 from test 10, no changes
