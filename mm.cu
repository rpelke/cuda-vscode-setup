// matmul_test.cu
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

// Naive SGEMM kernel: C = alpha * A @ B + beta * C
__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // Position in array C from a global perspective:
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < M && y < N) {
    float tmp = 0.0;
    for (int k = 0; k < K; ++k) {
      tmp += A[x * K + k] * B[k * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
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
    const int M = 70;
    const int N = 130;
    const int K = 65;
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

    // Block dimension
    dim3 blockDim(32, 32, 1);
    // Grid dimension
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);

    // Kernel launch
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference implementation of SGEMM
    cpu_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C_ref);

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
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
