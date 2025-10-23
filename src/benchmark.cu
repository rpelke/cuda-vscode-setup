#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "benchmark.cuh"
#include "sgemm/sgemm.cuh"
#include "utils.cuh"

void Benchmark::print_results(double ms, double gflops, std::string name) {
    std::cout << name << ": " << ms << " ms, " << gflops << " GFLOPS\n";
}

Benchmark::Benchmark() :
    d_A(nullptr), d_B(nullptr), d_C(nullptr), d_C_init_helper(nullptr) {}

// GEMM-like init
void Benchmark::init_matrices(std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C, std::vector<float> &h_C_init, std::vector<float> &h_C_cpu, std::vector<float> &h_C_cublas, int M, int K, int N) {
    h_A.resize(M * K);
    h_B.resize(K * N);
    h_C.resize(M * N);
    h_C_cpu.resize(M * N);
    h_C_cublas.resize(M * N);
    h_C_init.resize(M * N);

    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; ++i) {
        h_C_cpu[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C_init[i] = h_C_cpu[i];
        h_C_cublas[i] = 0.0f;
        h_C[i] = 0.0f;
    }
}

// GEMM-like init including cpu and cublas
void Benchmark::init_matrices(std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C, std::vector<float> &h_C_init, int M, int K, int N) {
    h_A.resize(M * K);
    h_B.resize(K * N);
    h_C.resize(M * N);
    h_C_init.resize(M * N);

    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; ++i) {
        h_C_init[i] = static_cast<float>(rand()) / RAND_MAX;;
        h_C[i] = 0.0f;
    }
}

// GEMV-like init
void Benchmark::init_matrices(std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C, std::vector<float> &h_C_init, int M, int N) {
    h_A.resize(M * N);
    h_B.resize(N);
    h_C.resize(M);
    h_C_init.resize(M);

    for (int i = 0; i < M * N; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M; ++i) {
        h_C_init[i] = static_cast<float>(rand()) / RAND_MAX;;
        h_C[i] = 0.0f;
    }
}

// Softmax-like init
void Benchmark::init_matrices(std::vector<float> &h_A, std::vector<float> &h_C, std::vector<float> &h_C_init, int M, int K) {
    h_A.resize(M * K);
    h_C.resize(M * K);
    h_C_init.resize(M * K);

    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * K; ++i) {
        h_C_init[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }
}

// Softmax-like init including cpu and cublas
void Benchmark::init_matrices(std::vector<float> &h_A, std::vector<float> &h_C, std::vector<float> &h_C_init, std::vector<float> &h_C_cpu, std::vector<float> &h_C_cublas, int M, int N) {
    h_A.resize(M * N);
    h_C.resize(M * N);
    h_C_init.resize(M * N);
    h_C_cpu.resize(M * N);
    h_C_cublas.resize(M * N);

    for (int i = 0; i < M * N; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; ++i) {
        h_C_init[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C_cpu[i] = h_C_init[i];
        h_C_cublas[i] = 0.0f;
        h_C[i] = 0.0f;
        h_C_cublas[i] = 0.0f;
    }
}

void Benchmark::copy_to_device(float *&d_A, float *&d_B, float *&d_C, float *&d_C_init_helper, std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C, std::vector<float> &res_vector, int M, int K, int N) {
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_init_helper, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_init_helper, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Benchmark::copy_to_device(float *&d_A, float *&d_B, float *&d_C, float *&d_C_init_helper, std::vector<float> &h_A, std::vector<float> &h_B, std::vector<float> &h_C, std::vector<float> &res_vector, int M, int N) {
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_init_helper, M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_init_helper, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Benchmark::copy_to_device(float *&d_A, float *&d_C, float *&d_C_init_helper, std::vector<float> &h_A, std::vector<float> &h_C, std::vector<float> &res_vector, int M, int N) {
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_init_helper, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_init_helper, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Benchmark::copy_results_to_host(float *&d_C, int M, int N,
                                     std::vector<float> &res_vector) {
    cudaMemcpy(res_vector.data(), d_C, res_vector.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void Benchmark::free_device_mem() {
    cudaFree(d_A);
    d_A = nullptr;
    cudaFree(d_B);
    d_B = nullptr;
    cudaFree(d_C);
    d_C = nullptr;
    cudaFree(d_C_init_helper);
    d_C_init_helper = nullptr;
}

bool Benchmark::validate_results(std::vector<float> &C_test, std::vector<float> &h_C_cpu,
                                 std::string test_name, int M, int N,
                                 float atol = 1e-2f) {
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float e = std::abs(C_test[i] - h_C_cpu[i]);
        if (e > atol) {
            ++mismatches;
            max_err = std::max(max_err, e);
        }
    }
    if (mismatches == 0) {
        std::cout << GREEN << "[" << test_name << ": TEST PASSED]\n" << RESET;
        return true;
    }
    std::cout << RED << test_name << ": TEST FAILED: " << mismatches
              << " mismatches, max error = " << max_err << "\n"
              << RESET;
    return false;
}

double Benchmark::benchmark_cublas(std::function<cublasStatus_t(cublasHandle_t)> cublasFunc, std::function<void(cudaStream_t)> resetResultTensor, int warmup, int iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasSetStream(handle, stream);

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < warmup; ++i) {
        resetResultTensor(stream);
        cublasFunc(handle);
    }

    double total_ms = 0.0;

    // Benchmarking
    for (int i = 0; i < iters; ++i) {
        resetResultTensor(stream);

        CUDA_CHECK(cudaEventRecord(start, stream));
        cublasFunc(handle);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cublasDestroy(handle);

    return total_ms / iters;
}