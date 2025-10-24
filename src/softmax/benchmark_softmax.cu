#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "benchmark.cuh"
#include "softmax/benchmark_softmax.cuh"
#include "softmax/softmax.cuh"
#include "utils.cuh"

SoftmaxBenchmark::SoftmaxBenchmark() :
    d_A(nullptr), d_B(nullptr), d_C(nullptr), d_C_init_helper(nullptr) {}

// Assuming full reuse of partial results
double SoftmaxBenchmark::ms_to_gflops(int M, int N, double ms) {
    int ops_per_exp = 15; // Approximate for __expf intrinsic, exp not an arithmetic function
    int exp_ops = M * N * ops_per_exp;
    // Binary sum, minimal
    int sum_ops = M * ceil(log2(N)); //sums
    int div_ops = M * N;
    double gflops = (exp_ops + sum_ops + div_ops) / (ms * 1e6);
    return gflops;
}

double SoftmaxBenchmark::benchmark_cpu(int M, int K) {
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting cpu calculation..." << std::endl;
    cpu_softmax(M, K, h_A, h_C_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(cpu_end - cpu_start)
        .count();
}

void SoftmaxBenchmark::benchmark_kernel(int M, int N,
                                 dim3 gridDim, dim3 blockDim,
                                 softmax_kernel_t launcher,
                                 std::string kernel_name, float atol = 1e-2f) {
    copy_to_device(d_A, d_C, d_C_init_helper, h_A, h_C, h_C_init, M, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launcher(M, N, d_A, d_C, gridDim, blockDim);
    cudaEventRecord(stop);

    cudaError_t err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
        printf("Kernel error: %s\n", cudaGetErrorString(err));

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    float kernel_gflops = ms_to_gflops(M, N, kernel_ms);
    copy_results_to_host(d_C, M, N, h_C);
    validate_results(h_C_cpu, h_C, kernel_name, M, N, atol);
    print_results(kernel_ms, kernel_gflops, kernel_name);
    free_device_mem(d_A, d_B, d_C, d_C_init_helper);
}

void SoftmaxBenchmark::benchmark_triple_softmax_kernel(int M, int N,
                                 dim3 gridDim_k0, dim3 gridDim_k1, dim3 gridDim_k2, dim3 blockDim, dim3 blockDim_k0, softmax_init_kernel_t k0, softmax_followUp_kernel_t k1, softmax_followUp_kernel_t k2, std::string kernel_name, float atol = 1e-2f) {
    copy_to_device(d_A, d_C, d_C_init_helper, h_A, h_C, h_C_init, M, N);

    // Init and copy temp matrix
    float *d_temp;
    std::vector<float> h_temp_init;
    h_temp_init.resize(M * gridDim_k0.x);
    for (int i = 0; i < M * gridDim_k0.x; ++i) {
        h_temp_init[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMalloc(&d_temp, M * gridDim_k0.x * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_temp, h_temp_init.data(),
                          h_temp_init.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    k0(M, N, d_A, d_C, d_temp, gridDim_k0, blockDim_k0);
    k1(M, N, d_A, d_C, d_temp, gridDim_k0.x, gridDim_k1, blockDim);
    k2(M, N, d_A, d_C, d_temp, gridDim_k0.x, gridDim_k2, blockDim);
    cudaEventRecord(stop);

    // Free temp matrix
    cudaFree(d_temp);

    cudaError_t err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
        printf("Kernel error: %s\n", cudaGetErrorString(err));

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    float kernel_gflops = ms_to_gflops(M, N, kernel_ms);
    copy_results_to_host(d_C, M, N, h_C);
    validate_results(h_C_cpu, h_C, kernel_name, M, N, 1e-2f);
    print_results(kernel_ms, kernel_gflops, kernel_name);
    free_device_mem(d_A, d_C, d_C_init_helper);
}

void SoftmaxBenchmark::benchmark_recursive_softmax_kernel(int M, int N, float atol = 1e-2f) {
    copy_to_device(d_A, d_C, d_C_init_helper, h_A, h_C, h_C_init, M, N);

    // Init and copy temp matrix
    float *d_temp;
    std::vector<float> h_temp_init;
    h_temp_init.resize(M * CEIL_DIV(N, 32));
    for (int i = 0; i < M * CEIL_DIV(N, 32); ++i) {
        h_temp_init[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMalloc(&d_temp, M * CEIL_DIV(N, 32) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_temp, h_temp_init.data(),
                          h_temp_init.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockDim_k0(32, 32, 1);
    dim3 gridDim_k0(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);
    dim3 gridDim_k1 = gridDim_k0;
    int N_1 = N;

    cudaEventRecord(start);
    softmax_warp_shuffle_k0<<<gridDim_k0, blockDim_k0>>>(M, N, d_A, d_C, d_temp);
    while (gridDim_k1.x > 1) {
        gridDim_k1.x = CEIL_DIV(gridDim_k1.x, 32);
        N_1 = CEIL_DIV(N_1, 32);
        softmax_warp_shuffle_k1<<<gridDim_k1, blockDim_k0>>>(M, N_1, d_A, d_C, d_temp, gridDim_k0.x);
    }
    softmax_block_binary_k2<<<gridDim_k0, blockDim_k0>>>(M, N, d_A, d_C, d_temp, gridDim_k0.x);
    cudaEventRecord(stop);

    // Free temp matrix
    cudaFree(d_temp);

    cudaError_t err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
        printf("Kernel error: %s\n", cudaGetErrorString(err));

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    float kernel_gflops = ms_to_gflops(M, N, kernel_ms);
    copy_results_to_host(d_C, M, N, h_C);
    validate_results(h_C_cpu, h_C, "Kernel 05", M, N, 1e-2f);
    print_results(kernel_ms, kernel_gflops, "Kernel 05");
    free_device_mem(d_A, d_C, d_C_init_helper);
}

void SoftmaxBenchmark::start_benchmarks(int M, int N) {
    // Initialize matrices
    init_matrices(h_A, h_C, h_C_init, h_C_cpu, h_C_cublas, M, N);

    // CPU reference
    double cpu_ms = benchmark_cpu(M, N);
    double cpu_gflops = ms_to_gflops(M, N, cpu_ms);
    print_results(cpu_ms, cpu_gflops, "CPU");

    // 00: Test simple softmax
    dim3 blockDim_00(BLOCKSIZE_00, BLOCKSIZE_00, 1);
    dim3 gridDim_00(CEIL_DIV(N, BLOCKSIZE_00), CEIL_DIV(M, BLOCKSIZE_00), 1);
    benchmark_kernel(
        M, N, gridDim_00, blockDim_00,
        [](int M, int N, const float *A, float *C, dim3 gridDim, dim3 blockDim) -> void {
            softmax_simple<<<gridDim, blockDim>>>(M, N, A, C);
        },
        "Kernel 00");

    // 01: Test softmax with partial summation
    benchmark_kernel(
        M, N, gridDim_00, blockDim_00,
        [](int M, int N, const float *A, float *C, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_sum<<<gridDim, blockDim>>>(M, N, A, C);
        },
        "Kernel 01");

    
    // 02: Test softmax with binary summation

    // Grid for k1 has only one column
    dim3 gridDim_k1(1, gridDim_00.y, gridDim_00.z);
    std::cout << "GridDim_k0: " << gridDim_00.x << ", " << gridDim_00.y << ", " << gridDim_00.z << std::endl;
    benchmark_triple_softmax_kernel(
        M, N, gridDim_00, gridDim_k1, gridDim_00, blockDim_00, blockDim_00,
        [](int M, int N, const float *A, float *C, float *temp, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k0<<<gridDim, blockDim>>>(M, N, A, C, temp);
        },
        [](int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k1<<<gridDim, blockDim, gridDim_y_k0*BLOCKSIZE_00 * sizeof(float)>>>(M, N, A, C, temp, gridDim_y_k0);
        },
        [](int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k2<<<gridDim, blockDim>>>(M, N, A, C, temp, gridDim_y_k0);
        },
        "Kernel 02"
    );

    // 03: Test softmax with resolved warp divergencies
    dim3 gridDim_k0(CEIL_DIV(M, 2*BLOCKSIZE_00), CEIL_DIV(N, BLOCKSIZE_00), 1);
    benchmark_triple_softmax_kernel(
        M, N, gridDim_k0, gridDim_k1, gridDim_00, blockDim_00, blockDim_00,
        [](int M, int N, const float *A, float *C, float *temp, dim3 gridDim, dim3 blockDim) -> void {
            softmax_binary_non_divergent_k0<<<gridDim, blockDim>>>(M, N, A, C, temp);
        },
        [](int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_binary_non_divergent_k1<<<gridDim, blockDim, gridDim_y_k0*BLOCKSIZE_00 * sizeof(float)>>>(M, N, A, C, temp, gridDim_y_k0);
        },
        [](int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k2<<<gridDim, blockDim>>>(M, N, A, C, temp, gridDim_y_k0);
        },
        "Kernel 03"
    );

    // 04: Test softmax with warp shuffle
    benchmark_triple_softmax_kernel(
        M, N, gridDim_00, gridDim_k1, gridDim_00, blockDim_00, blockDim_00,
        [](int M, int N, const float *A, float *C, float *temp, dim3 gridDim, dim3 blockDim) -> void {
            softmax_warp_shuffle_k0<<<gridDim, blockDim>>>(M, N, A, C, temp);
        },
        [](int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_binary_non_divergent_k1<<<gridDim, blockDim, gridDim_y_k0*BLOCKSIZE_00 * sizeof(float)>>>(M, N, A, C, temp, gridDim_y_k0);
        },
        [](int M, int N, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k2<<<gridDim, blockDim>>>(M, N, A, C, temp, gridDim_y_k0);
        },
        "Kernel 04"
    );

    benchmark_recursive_softmax_kernel(M, N);
}