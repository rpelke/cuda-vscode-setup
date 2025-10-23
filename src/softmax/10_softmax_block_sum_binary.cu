#include "softmax/softmax.cuh"

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

    int globalStartElem = y*N+x+threadIdx.x;
    int localStartElem = threadIdx.y * blockDim.x + 2*threadIdx.x;

    // load A into shared mem
    __shared__ float As[BLOCKSIZE_10 * BLOCKSIZE_10];

    // Threads that exceed block should idle
    if(2*threadIdx.x >= blockDim.x) return;

    // Each thread loads it's start element and it's neighbour (if exists)
    As[localStartElem] = A[globalStartElem];
    if(2*threadIdx.x+1 < blockDim.x)
        As[localStartElem + 1] = A[globalStartElem+1];

    __syncthreads();

    // Init elements for softmax
    As[localStartElem] = __expf(As[localStartElem]);

    __syncthreads();

    // Each thread calculates a partial sum
    int i = 1;
    for (; i <= blockDim.x; i*=2) {
        bool outOfBlock = 2*threadIdx.x+i >= blockDim.x;
        bool outOfMat = x+threadIdx.x+i >= N;
        if(threadIdx.x % i == 0 && !outOfBlock && !outOfMat) {
            if (i == 1) As[localStartElem] += __expf(As[localStartElem + i]);
            else As[localStartElem] += As[localStartElem + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
        temp[y * gridDim.x + blockIdx.x] = As[localStartElem];

}

// Kernel 1 does reduction across blocks (use a single y-block for this)
__global__ void softmax_block_binary_k1(int M, int N, const float *A, float *C, float *temp, int gridDim_x_k0) {

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
        if(threadIdx.x % i == 0 && 2*threadIdx.x+i < gridDim_x_k0) {
            temp_s[localStartElem] += temp_s[localStartElem + i];
        }
        __syncthreads();
    }

    // Write result back to temp
    if(threadIdx.x == 0)
        temp[globalStartElem] = temp_s[localStartElem];
}

// Kernel 2 does final calculation on reduced values
__global__ void softmax_block_binary_k2(int M, int N, const float *A, float *C, float *temp, int gridDim_x_k0) {

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Only 1 op on data, GMEM access should be fine
    C[y * N + x] = expf(A[y * N + x]) / temp[y*gridDim_x_k0];
}