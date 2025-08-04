// matmul_test.cu
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

constexpr int BLOCKSIZE = 32;
constexpr int CEIL_DIV(int a, int b) { return ((a) + (b) - 1) / (b); }

// Tiled SGEMM kernel: C = alpha * A @ B + beta * C
__global__ void sgemm_tiled(int M, int N, int K,
                            float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // Block's positiion inside C
    int b_y = blockIdx.y;
    int b_x = blockIdx.x;

    // Thread's position inside the block (tile) 
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;

    // Pointers to the block's top-left (position in C)
    const int C_tile_offs = (BLOCKSIZE * N * b_y) + (BLOCKSIZE * b_x);
    // Offset to row=0 and col=b_x in B
    const int B_tile_offs = BLOCKSIZE * b_x;
    // Offset to row=b_y and col=0 in A
    const int A_tile_offs = (BLOCKSIZE * K) * b_y;

    // Shared-memory buffers for A and B tiles
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    float tmp = 0.0f;
    
    // k={0,31,63,...}
    for (int k=0; k<K; k+=BLOCKSIZE) {
        // Load one A value and one B value per thread to shared memory
        if ((k + t_x < K) && (b_y * BLOCKSIZE + t_y < M)) {
            As[BLOCKSIZE * t_y + t_x] = A[A_tile_offs
                + K * t_y
                + k + t_x
            ];
        } else {
            As[BLOCKSIZE * t_y + t_x] = 0.0f; // Out of bounds
        }
        
        if ((b_x * BLOCKSIZE + t_x < N) && (k + t_y < K)) {
            Bs[BLOCKSIZE * t_y + t_x] = B[
                + N * k
                + N * t_y
                + B_tile_offs + t_x
            ];
        } else {
            Bs[BLOCKSIZE * t_y + t_x] = 0.0f; // Out of bounds
        }
        
        // Block threads to wait for all threads to load their values
        __syncthreads();

        // Execute dot product for the current cached block
        for (int i=0; i<BLOCKSIZE; ++i) {
            tmp += As[BLOCKSIZE * t_y + i] * Bs[BLOCKSIZE * i + t_x];
        }
        
        __syncthreads();
    }

    if ((b_x * BLOCKSIZE + t_x < N) && (b_y * BLOCKSIZE + t_y < M)) {
        C[C_tile_offs + N * t_y + t_x] =
            alpha * tmp + beta * C[C_tile_offs + N * t_y + t_x];
    }
}

// CPU reference implementation of SGEMM
void cpu_sgemm(int M, int N, int K,
               float alpha, const std::vector<float> &A,
               const std::vector<float> &B, float beta, std::vector<float> &C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k)
                tmp += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * tmp + beta * C[i * N + j];
        }
    }
}

int main() {
    // Dimensions
    const int M = 99;
    const int N = 555;
    const int K = 971;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Host matrices
    std::vector<float> h_A(M*K), h_B(K*N), h_C(M*N), h_C_ref(M*N);

    // Initialization
    for (int i = 0; i < M*K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K*N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M*N; ++i) {
        h_C[i]     = static_cast<float>(rand()) / RAND_MAX;
        h_C_ref[i] = h_C[i];
    }

    // Device pointer
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    // Data transfer: Host → Device
    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridDim(
        CEIL_DIV(N, BLOCKSIZE),
        CEIL_DIV(M, BLOCKSIZE),
        1
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel launch
    cudaEventRecord(start);
    sgemm_tiled<<<gridDim, blockDim>>>(
        M, N, K,
        alpha,
        d_A,
        d_B,
        beta,
        d_C
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference implementation of SGEMM
    cpu_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C_ref);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);  // elapsed time in milliseconds
    double time_sec = ms / 1e3;

    double tops = 2.0 * M * N * K / (time_sec * 1e12);
    std::cout << "GPU SGEMM: " << ms << " ms, "
              << tops << " TOPS\n";

    // Comparison & error analysis
    int   mismatches = 0;
    float max_err    = 0.0f;
    for (int i = 0; i < M*N; ++i) {
        float err = std::abs(h_C_ref[i] - h_C[i]);
        if (err > 1e-3f) {
            ++mismatches;
            if (err > max_err) max_err = err;
        }
    }

    if (mismatches == 0) {
        std::cout << "TEST PASSED\n";
    } else {
        std::cout << "TEST FAILED: " << mismatches
                  << " Nicht-Übereinstimmungen, max Fehler = "
                  << max_err << "\n";
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
