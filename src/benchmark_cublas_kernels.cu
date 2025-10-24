#include "benchmark.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "utils.cuh"

void free_sgemm_matrices(float *&d_A, float *&d_B, float *&d_C, float *&d_C_init_helper){
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_init_helper);
    d_A = nullptr;
    d_B = nullptr;
    d_C = nullptr;
    d_C_init_helper = nullptr;
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_init_helper = nullptr;
    float alpha = 2.0f;
    float beta = 3.0f;

    Benchmark b;

    std::vector<float> h_A, h_B, h_C, res;
    b.init_matrices(h_A, h_B, h_C, res, M, N, K);
    b.copy_to_device(d_A, d_B, d_C, d_C_init_helper, h_A, h_B, h_C, res, M, N, K);

    // Benchmark all gemmEx algorithms
    std::function<void(cudaStream_t)> resetC = [=](cudaStream_t stream) {CUDA_CHECK(cudaMemcpyAsync(d_C, d_C_init_helper, M * N * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));};

    for (int i = CUBLAS_GEMM_DEFAULT; i <= CUBLAS_GEMM_ALGO23 ; i++) {
        cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(i);

        std::function<cublasStatus_t(cublasHandle_t)> kernel = [=](cublasHandle_t handle) -> cublasStatus_t { return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B,
                     CUDA_R_32F, N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F,
                     N, CUDA_R_32F, algo);};

        double cublas_ms = b.benchmark_cublas(kernel, resetC);
        std::cout << "SGEMM Algorithm " << i << " finished in " << cublas_ms << " ms." << std::endl;
    }
    for (int i = CUBLAS_GEMM_DEFAULT_TENSOR_OP; i <= CUBLAS_GEMM_ALGO15_TENSOR_OP ; i++) {
        cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(i);

        std::function<cublasStatus_t(cublasHandle_t)> kernel = [=](cublasHandle_t handle) -> cublasStatus_t { return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B,
                     CUDA_R_32F, N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F,
                     N, CUDA_R_32F, algo);};

        double cublas_ms = b.benchmark_cublas(kernel, resetC);
        std::cout << "SGEMM Algorithm " << i << " finished in " << cublas_ms << " ms." << std::endl;
    }

    free_sgemm_matrices(d_A, d_B, d_C, d_C_init_helper);

    std::cout << "-------------------------" << std::endl;

    b.init_matrices(h_A, h_B, h_C, res, M, N);
    b.copy_to_device(d_A, d_B, d_C, d_C_init_helper, h_A, h_B, h_C, res, M, N);

    std::function<cublasStatus_t(cublasHandle_t)> kernel = [=](cublasHandle_t handle) -> cublasStatus_t { return cublasSgemv_v2(handle, CUBLAS_OP_N, M, N,
                   &alpha, d_A, M, d_B,
                   1, &beta, d_C, 1);};

    std::function<void(cudaStream_t)> resetC2 = [=](cudaStream_t stream) {CUDA_CHECK(cudaMemcpyAsync(d_C, d_C_init_helper, M * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));};

    double cublas_ms = b.benchmark_cublas(kernel, resetC2);
    std::cout << "SGEMV finished in " << cublas_ms << " ms." << std::endl;

    free_sgemm_matrices(d_A, d_B, d_C, d_C_init_helper);
}