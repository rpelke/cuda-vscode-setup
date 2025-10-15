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

__global__ void softmax_block_binary_k0(int M, int N, const float *A, float *C, float *temp) { // temp holds results for across-block reduction
    // Position in array C from a global perspective:

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    int globalStartElem = x*N+y+threadIdx.y;
    int localStartElem = threadIdx.x * blockDim.y + 2*threadIdx.y;

    // load A into shared mem
    __shared__ float As[BLOCKSIZE_10 * BLOCKSIZE_10];

    // Threads that exceed block should idle
    if(2*threadIdx.y >= blockDim.y) return;

    // Each thread loads it's start element and it's neighbour (if exists)
    As[localStartElem] = A[globalStartElem];
    if(2*threadIdx.y+1 < blockDim.y)
        As[localStartElem + 1] = A[globalStartElem+1];

    __syncthreads();

    // Init elements for softmax
    As[localStartElem] = expf(As[localStartElem]);

    __syncthreads();

    // Each thread calculates a partial sum
    int i = 1;
    for (; i < blockDim.y; i*=2) {
        if(threadIdx.y % i == 0 && localStartElem+i < blockDim.y) {
            if (i == 1) As[localStartElem] += expf(As[localStartElem + i]);
            else As[localStartElem] += As[localStartElem + i];
        }
        __syncthreads();
    }
    // Handle leftover (additional add necessary if blockDim.y not a power of 2)
    if (i != blockDim.y) {
        if(threadIdx.y % i == 0) {
            if (i == 1) As[localStartElem] += expf(As[localStartElem + i]);
            else As[localStartElem] += As[localStartElem + i];
        }
        __syncthreads();
    }

    // TODO: Reduction across blocks (global sync not possible, write to global mem, launch another kernel)

    if(threadIdx.y == 0)
        temp[(blockIdx.x * blockDim.x + threadIdx.x) * gridDim.y + blockIdx.y] = As[localStartElem];

}

// Kernel 1 does reduction across blocks (use a single y-block for this)
__global__ void softmax_block_binary_k1(int M, int N, const float *A, float *C, float *temp) {

    // load temp into shared mem (split along x-axis)
    __shared__ float temp_s[BLOCKSIZE_10 * BLOCKSIZE_10]; // Ensure blockDim.y is equals gridSize.y from k0

    // Threads that exceed block should idle
    if(2*threadIdx.y >= blockDim.y) return;

    // Each thread loads it's start element and it's neighbour (if exists)
    int startElem = threadIdx.x * blockDim.y + 2*threadIdx.y;

    temp_s[startElem] = temp[startElem];
    if(threadIdx.y+1 < blockDim.y)
        temp_s[startElem + 1] = temp[startElem+1];

    __syncthreads();

    // Each thread calculates a partial sum
    int i = 1;
    for (; i < blockDim.y; i*=2) {
    //for (; i <= 4; i*=2) {
        if(threadIdx.y % i == 0) {
            temp_s[startElem] += temp_s[startElem + i];
        }
        __syncthreads();
    }

    // Handle leftover (additional adds necessary if blockDim.y not a power of 2)
    if (i != blockDim.y) {
        if(threadIdx.y % i == 0) {
            temp_s[startElem] += temp_s[startElem + i];
        }
        __syncthreads();
    }

    // Write result back to temp
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        temp[0] = temp_s[0];
    }
    //if(threadIdx.y == 0)
        //temp[(blockIdx.x * blockDim.x + threadIdx.x) * blockDim.y] = temp_s[threadIdx.x*blockDim.y];
}

// Kernel 2 does final calculation on reduced values
__global__ void softmax_block_binary_k2(int M, int N, const float *A, float *C, float *temp) {

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Only 1 op on data, GMEM access should be fine
    C[x * N + y] = expf(A[x * N + y]) / temp[threadIdx.x];
}