#include "softmax.cuh"

// This impl does partial sums, but as a binary tree to reduce calculations

// Kernel 0 does across-thread reduction and writes results to GMEM
// As stores entire block
// temp: ________________________
//       | b_x0_y0_s0 b_x0_y1_s0
//       | b_x0_y0_s1 b_x0_y1_s1
//       | ...
//       | b_x1_y0_s0 b_x1_y1_s0
//       | b_x1_y0_s1 b_x1_y1_s1

__global__ void softmax_binary_non_divergent_k0(int M, int N, const float *A, float *C, float *temp) { // temp holds results for across-block reduction
    // Position in array C from a global perspective:

    const unsigned int x = threadIdx.x + blockIdx.x * BLOCKSIZE_X_11;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    int globalStartElem = y*N+x+threadIdx.x;
    int localStartElem = threadIdx.y * BLOCKSIZE_X_11 + 2*threadIdx.x;

    // load A into shared mem
    __shared__ float As[BLOCKSIZE_11 * BLOCKSIZE_X_11];

    // Threads that exceed block should idle
    if(2*threadIdx.x >= BLOCKSIZE_X_11) return;

    // Each thread loads it's start element and it's neighbour (if exists)
    As[localStartElem] = __expf(A[globalStartElem]);
    if(2*threadIdx.x+1 < BLOCKSIZE_X_11)
        As[localStartElem + 1] = A[globalStartElem+1];

    __syncthreads();

    // Each thread calculates a partial sum
    int i = 1;
    for (; i <= BLOCKSIZE_X_11; i*=2) {
        int index = 2*i*threadIdx.x;
        bool outOfBlock = index+i >= BLOCKSIZE_X_11;
        bool outOfMat = blockIdx.x*BLOCKSIZE_X_11+index+i >= N;

        if(!outOfBlock && !outOfMat) {
            if (i == 1) As[threadIdx.y*BLOCKSIZE_X_11+index] += __expf(As[threadIdx.y*BLOCKSIZE_X_11+index + i]);
            else As[threadIdx.y*BLOCKSIZE_X_11+index] += As[threadIdx.y*BLOCKSIZE_X_11+index + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
        temp[y * gridDim.x + blockIdx.x] = As[localStartElem];
}

// Kernel 1 does reduction across blocks (use a single y-block for this)
__global__ void softmax_binary_non_divergent_k1(int M, int N, const float *A, float *C, float *temp, int gridDim_x_k0) {

    // load temp into shared mem (split along x-axis)
    extern __shared__ float temp_s[]; // BLOCKSIZE_10*gridDim_x_k0

    // Threads that exceed block should idle
    if(2*threadIdx.x >= gridDim_x_k0) return;

    // Each thread loads it's start element and it's neighbour (if exists)
    int globalStartElem = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim_x_k0 + 2*threadIdx.x;
    int localStartElem = threadIdx.y * gridDim_x_k0 + 2*threadIdx.x;

    temp_s[localStartElem] = temp[globalStartElem];
    if(threadIdx.x+1 < gridDim_x_k0)
        temp_s[localStartElem + 1] = temp[globalStartElem+1];

    __syncthreads();

    // Each thread calculates a partial sum
    int i = 1;
    for (; i <= gridDim_x_k0; i*=2) {
        int index = 2*i*threadIdx.x;
        bool outOfMat = index+i >= gridDim_x_k0;

        if(!outOfMat) {
            temp_s[threadIdx.y * gridDim_x_k0+index] += temp_s[threadIdx.y * gridDim_x_k0 + index + i];
        }
        __syncthreads();
    }

    // Write result back to temp
    if(threadIdx.x == 0)
        temp[globalStartElem] = temp_s[localStartElem];
}

// Use k2 from test 10, no changes