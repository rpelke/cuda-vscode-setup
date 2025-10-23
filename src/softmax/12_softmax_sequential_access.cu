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

__device__ constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

__global__ void softmax_sequential_access_k0(int M, int N, const float *A, float *C, float *temp) { // temp holds results for across-block reduction
    // Position in array C from a global perspective:

    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * BLOCKSIZE_10;

    int globalStartElem = x*N+y+threadIdx.y;
    int localStartElem = threadIdx.x * BLOCKSIZE_10 + 2*threadIdx.y;

    // load A into shared mem
    __shared__ float As[BLOCKSIZE_10 * BLOCKSIZE_10];

    // Threads that exceed block should idle
    if(2*threadIdx.y >= BLOCKSIZE_10) return;

    // Each thread loads it's start element and it's neighbour (if exists)
    As[localStartElem] = A[globalStartElem];
    if(2*threadIdx.y+1 < BLOCKSIZE_10)
        As[localStartElem + 1] = A[globalStartElem+1];

    __syncthreads();

    // Init elements for softmax
    As[localStartElem] = __expf(As[localStartElem]);

    __syncthreads();

    // Each thread calculates a partial sum
    int i = blockDim.y;
    //for (; i > 1; i=(i==1)?0:CEIL_DIV(i, 2)) {
    for (; i > 1; i=0) {
        bool outOfBlock = threadIdx.y+i >= BLOCKSIZE_10;
        bool outOfMat = threadIdx.x*BLOCKSIZE_10+threadIdx.y+i >= N;

        if(threadIdx.y < i && !outOfBlock && !outOfMat) {
            if (i == 1) As[threadIdx.x*BLOCKSIZE_10+threadIdx.y] += __expf(As[threadIdx.x*BLOCKSIZE_10+threadIdx.y+i]);
            else As[threadIdx.x*BLOCKSIZE_10+threadIdx.y] += As[threadIdx.x*BLOCKSIZE_10+threadIdx.y+i];
        }
        __syncthreads();
    }

    if(threadIdx.y == 0)
        temp[x * gridDim.y + blockIdx.y] = As[localStartElem];
}

// Kernel 1 does reduction across blocks (use a single y-block for this)
__global__ void softmax_sequential_access_k1(int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0) {

    // load temp into shared mem (split along x-axis)
    extern __shared__ float temp_s[]; // BLOCKSIZE_10*gridDim_y_k0

    // Threads that exceed block should idle
    if(2*threadIdx.y >= gridDim_y_k0) return;

    // Each thread loads it's start element and it's neighbour (if exists)
    int globalStartElem = (blockIdx.x * blockDim.x + threadIdx.x) * gridDim_y_k0 + 2*threadIdx.y;
    int localStartElem = threadIdx.x * gridDim_y_k0 + 2*threadIdx.y;

    temp_s[localStartElem] = temp[globalStartElem];
    if(threadIdx.y+1 < gridDim_y_k0)
        temp_s[localStartElem + 1] = temp[globalStartElem+1];

    __syncthreads();

    // Each thread calculates a partial sum
    int i = 1;
    for (; i <= gridDim_y_k0; i*=2) {
        int index = 2*i*threadIdx.y;
        bool outOfMat = index+i >= gridDim_y_k0;

        if(!outOfMat) {
            temp_s[threadIdx.x * gridDim_y_k0+index] += temp_s[threadIdx.x * gridDim_y_k0 + index + i];
        }
        __syncthreads();
    }

    // Write result back to temp
    if(threadIdx.y == 0)
        temp[globalStartElem] = temp_s[localStartElem];
}

// Use k2 from test 10, no changes