#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "benchmark.cuh"
#include "sgemm.cuh"
#include "softmax.cuh"

#define CUDA_CHECK(val) cudaCheck((val), __FILE__, __LINE__)
inline void cudaCheck(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " -> "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void print_results(double ms, double gflops, std::string name) {
    std::cout << name << ": " << ms << " ms, " << gflops << " GFLOPS\n";
}

Benchmark::Benchmark() : d_A(nullptr), d_B(nullptr), d_C(nullptr) {}

void Benchmark::init_matrices(int M, int K, int N) {
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

void Benchmark::init_matrices(int M, int K) {
    h_A.resize(M * K);
    h_C.resize(M * K);
    h_C_cpu.resize(M * K);
    h_C_init.resize(M * K);

    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * K; ++i) {
        h_C_cpu[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C_init[i] = h_C_cpu[i];
        h_C[i] = 0.0f;
    }
}

void Benchmark::copy_to_device(int M, int K, int N,
                               std::vector<float> &res_vector) {
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Benchmark::copy_to_device(int M, int K,
                               std::vector<float> &res_vector) {
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, res_vector.data(),
                          res_vector.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Benchmark::copy_results_to_host(int M, int N,
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
}

bool Benchmark::validate_results(std::vector<float> &C_test,
                                 std::string test_name, int M, int N,
                                 float atol = 1e-2f) {
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        //std::cout << C_test[i] << ", " << h_C_cpu[i] << std::endl;
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

double Benchmark::benchmark_cpu(int M, int K, int N, float alpha, float beta) {
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(cpu_end - cpu_start)
        .count();
}

double Benchmark::benchmark_softmax_cpu(int M, int K) {
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting cpu calculation..." << std::endl;
    cpu_softmax(M, K, h_A, h_C_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(cpu_end - cpu_start)
        .count();
}

double Benchmark::ms_to_gflops(int M, int K, int N, double ms) {
    double gflops = 2.0 * M * N * K / (ms * 1e6);
    return gflops;
}

// Assuming full reuse of partial results
double Benchmark::ms_to_gflops(int M, int N, double ms) {
    int sum_ops = M * N + M * (N-1); // exp calls + sums
    int div_ops = M * N;
    double gflops = (sum_ops + div_ops) / (ms * 1e6);
    return gflops;
}

double Benchmark::benchmark_cublas(int M, int K, int N, float alpha,
                                   float beta) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B,
                 CUDA_R_32F, N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cublasDestroy(handle);
    float cublas_ms;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    return cublas_ms;
}

void Benchmark::benchmark_kernel(int M, int K, int N, float alpha, float beta,
                                 dim3 gridDim, dim3 blockDim,
                                 sgemm_kernel_t launcher,
                                 std::string kernel_name, float atol = 1e-2f) {
    copy_to_device(M, K, N, h_C_init);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launcher(M, N, K, alpha, d_A, d_B, beta, d_C, gridDim, blockDim);
    cudaEventRecord(stop);

    cudaError_t err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
        printf("Kernel error: %s\n", cudaGetErrorString(err));

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    float kernel_gflops = ms_to_gflops(M, K, N, kernel_ms);
    copy_results_to_host(M, N, h_C);
    validate_results(h_C, kernel_name, M, N, atol);
    print_results(kernel_ms, kernel_gflops, kernel_name);
    free_device_mem();
}

void Benchmark::benchmark_softmax_kernel(int M, int K,
                                 dim3 gridDim, dim3 blockDim,
                                 softmax_kernel_t launcher,
                                 std::string kernel_name, float atol = 1e-2f) {
    copy_to_device(M, K, h_C_init);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launcher(M, K, d_A, d_C, gridDim, blockDim);
    cudaEventRecord(stop);

    cudaError_t err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
        printf("Kernel error: %s\n", cudaGetErrorString(err));

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    float kernel_gflops = ms_to_gflops(M, K, kernel_ms);
    copy_results_to_host(M, K, h_C);
    validate_results(h_C, kernel_name, M, K, atol);
    print_results(kernel_ms, kernel_gflops, kernel_name);
    free_device_mem();
}

void Benchmark::benchmark_triple_softmax_kernel(int M, int K,
                                 dim3 gridDim, dim3 gridDim_k1, dim3 blockDim, dim3 blockDim_k0, softmax_init_kernel_t k0, softmax_followUp_kernel_t k1, softmax_followUp_kernel_t k2, std::string kernel_name, float atol = 1e-2f) {
    copy_to_device(M, K, h_C_init);

    // Init and copy temp matrix
    float *d_temp;
    std::vector<float> h_temp_init;
    h_temp_init.resize(M * gridDim.y);
    for (int i = 0; i < M * gridDim.y; ++i) {
        h_temp_init[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMalloc(&d_temp, M * gridDim.y * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_temp, h_temp_init.data(),
                          h_temp_init.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*std::cout << "A: " << std::endl;
    for (int i = 1024-32; i < 1024; i++) {
        std::cout << h_A[i] << ", ";
    }
    std::cout << std::endl;*/

    cudaEventRecord(start);
    k0(M, K, d_A, d_C, d_temp, gridDim, blockDim_k0);
    k1(M, K, d_A, d_C, d_temp, gridDim.y, gridDim_k1, blockDim);
    k2(M, K, d_A, d_C, d_temp, gridDim.y, gridDim, blockDim);
    cudaEventRecord(stop);

    // test transfer back
    cudaMemcpy(h_temp_init.data(), d_temp, h_temp_init.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "Test: " << std::endl;
    for (int i = 0; i < 32; i++) {
        //std::cout << h_temp_init[i*gridDim.y] << ", ";
        std::cout << h_temp_init[i] << ", ";
    }
    std::cout << std::endl;

    // Free temp matrix
    cudaFree(d_temp);

    cudaError_t err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
        printf("Kernel error: %s\n", cudaGetErrorString(err));

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    float kernel_gflops = ms_to_gflops(M, K, kernel_ms);
    copy_results_to_host(M, K, h_C);
    validate_results(h_C, kernel_name, M, K, 1e-2f);
    print_results(kernel_ms, kernel_gflops, kernel_name);
    free_device_mem();
}

void Benchmark::start_benchmarks(int M, int K, int N, float alpha, float beta) {
    // Initialize matrices
    init_matrices(M, K, N);

    // CPU reference
    double cpu_ms = benchmark_cpu(M, K, N, alpha, beta);
    double cpu_gflops = ms_to_gflops(M, K, N, cpu_ms);
    print_results(cpu_ms, cpu_gflops, "CPU");

    // Cublas reference
    copy_to_device(M, K, N, h_C_init);
    double cublas_ms = benchmark_cublas(M, K, N, alpha, beta);
    double cublas_gflops = ms_to_gflops(M, K, N, cublas_ms);
    copy_results_to_host(M, N, h_C_cublas);
    validate_results(h_C_cublas, "Cublas", M, N, 5e-2f);
    print_results(cublas_ms, cublas_gflops, "Cublas");
    free_device_mem();

    // 00: Test simple kernel
    dim3 blockDim_00(BLOCKSIZE_00, BLOCKSIZE_00, 1);
    dim3 gridDim_00(CEIL_DIV(M, BLOCKSIZE_00), CEIL_DIV(N, BLOCKSIZE_00), 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_00, blockDim_00,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) -> void {
            sgemm_simple<<<gridDim, blockDim>>>(M, K, N, alpha, A, B, beta, C);
        },
        "Kernel 00");

    // 01: Test coalesced kernel
    dim3 blockDim_01(BLOCKSIZE_01 * BLOCKSIZE_01);
    dim3 gridDim_01(CEIL_DIV(N, BLOCKSIZE_01), CEIL_DIV(M, BLOCKSIZE_01), 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_01, blockDim_01,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) -> void {
            sgemm_coalesced<<<gridDim, blockDim>>>(M, K, N, alpha, A, B, beta,
                                                   C);
        },
        "Kernel 01");

    // 02: Test tiled kernel
    dim3 blockDim_02(BLOCKSIZE_02, BLOCKSIZE_02, 1);
    dim3 gridDim_02(CEIL_DIV(N, BLOCKSIZE_02), CEIL_DIV(M, BLOCKSIZE_02), 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_02, blockDim_02,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) -> void {
            sgemm_tiled<<<gridDim, blockDim>>>(M, K, N, alpha, A, B, beta, C);
        },
        "Kernel 02");

    // 03: Test tiled 2D kernel
    static_assert(BN_03 % TN_03 == 0 && BM_03 % TM_03 == 0,
                  "BN % TN != 0 || BM % TM != 0");
    static_assert(BN_03 / TN_03 == BK_03, "BN / TN != BK");
    static_assert(BM_03 / TM_03 == BK_03, "BM / TM != BK");
    static_assert(BK_03 >= TM_03 && BK_03 >= TN_03, "BK < TM || BK < TN");
    dim3 gridDim_03(CEIL_DIV(N, BN_03), CEIL_DIV(M, BM_03), 1);
    dim3 blockDim_03(BN_03 / TN_03, BM_03 / TM_03, 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_03, blockDim_03,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) {
            sgemm_tiled_2d<<<gridDim, blockDim>>>(M, K, N, alpha, A, B, beta,
                                                  C);
        },
        "Kernel 03");

    // 04: Test tiled 2D kernel with vectorization
    static_assert(BN_04 % TN_04 == 0 && BM_04 % TM_04 == 0,
                  "BN % TN != 0 || BM % TM != 0");
    static_assert(BN_04 / TN_04 == BK_04, "BN / TN != BK");
    static_assert(BM_04 / TM_04 == BK_04, "BM / TM != BK");
    static_assert(BK_04 >= TM_04 && BK_04 >= TN_04, "BK < TM || BK < TN");
    static_assert(BK_04 >= VEC_SIZE_04 && BK_04 % VEC_SIZE_04 == 0,
                  "BK < VEC_SIZE_04 || BK % VEC_SIZE_04 != 0");
    dim3 gridDim_04(CEIL_DIV(N, BN_04), CEIL_DIV(M, BM_04), 1);
    dim3 blockDim_04(BN_04 / TN_04, BM_04 / TM_04, 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_04, blockDim_04,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) {
            sgemm_tiled_2d_vectorized_1<<<gridDim, blockDim>>>(M, K, N, alpha,
                                                               A, B, beta, C);
        },
        "Kernel 04");

    // 05: Test tiled 2D kernel with vectorization
    static_assert(BN_05 % TN_05 == 0 && BM_05 % TM_05 == 0,
                  "BN % TN != 0 || BM % TM != 0");
    static_assert(BN_05 / TN_05 == BK_05, "BN / TN != BK");
    static_assert(BM_05 / TM_05 == BK_05, "BM / TM != BK");
    static_assert(BK_05 >= TM_05 && BK_05 >= TN_05, "BK < TM || BK < TN");
    static_assert(BK_05 >= VEC_SIZE_05 && BK_05 % VEC_SIZE_05 == 0,
                  "BK < VEC_SIZE_05 || BK % VEC_SIZE_05 != 0");
    static_assert((TN_05 >= VEC_SIZE_05) && (TN_05 % VEC_SIZE_05) == 0,
                  "TN < VEC_SIZE_05 || TN % VEC_SIZE_05 != 0");
    dim3 gridDim_05(CEIL_DIV(N, BN_05), CEIL_DIV(M, BM_05), 1);
    dim3 blockDim_05(BN_05 / TN_05, BM_05 / TM_05, 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_05, blockDim_05,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) {
            sgemm_tiled_2d_vectorized_2<<<gridDim, blockDim>>>(M, K, N, alpha,
                                                               A, B, beta, C);
        },
        "Kernel 05");

    // 06: Test warptiling
    static_assert(BN_06 % WN_06 == 0 && BM_06 % WM_06 == 0,
                  "BN % WN != 0 || BM % WM != 0");
    static_assert(BN_06 / TN_06 == BK_06, "BN / TN != BK");
    static_assert(BM_06 / TM_06 == BK_06, "BM / TM != BK");
    static_assert(BK_06 >= TM_06 && BK_06 >= TN_06, "BK < TM || BK < TN");
    static_assert(WN_06 >= TN_06 && WM_06 >= TM_06, "WN < TN || WM < TM");
    static_assert(WN_06 % TN_06 == 0 && WM_06 % TM_06 == 0,
                  "WN % TN != 0 || WM % TM != 0");
    dim3 gridDim_06(CEIL_DIV(N, BN_06), CEIL_DIV(M, BM_06), 1);
    dim3 blockDim_06(BN_06 / TN_06, BM_06 / TM_06, 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_06, blockDim_06,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) {
            sgemm_warptiling<<<gridDim, blockDim>>>(M, K, N, alpha, A, B, beta,
                                                    C);
        },
        "Kernel 06");

    // 07: Test tensor cores
    static_assert(TN_07 * TM_07 * 32 == BLOCKSIZE_07 * BLOCKSIZE_07);
    dim3 gridDim_07(CEIL_DIV(N, BLOCKSIZE_07), CEIL_DIV(M, BLOCKSIZE_07), 1);
    dim3 blockDim_07(BLOCKSIZE_07 / TN_07, BLOCKSIZE_07 / TM_07, 1);
    benchmark_kernel(
        M, K, N, alpha, beta, gridDim_07, blockDim_07,
        [](int M, int K, int N, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 gridDim, dim3 blockDim) {
            sgemm_tensorcores<<<gridDim, blockDim>>>(M, K, N, alpha, A, B, beta,
                                                     C);
        },
        "Kernel 07", 1e-1f /*Higher tolerance due to fp32->fp16 conversion*/);
}

void Benchmark::start_softmax_benchmarks(int M, int K) {
    std::cout << "Starting softmax benchmarks" << std::endl;
    // Initialize matrices
    init_matrices(M, K);
    std::cout << "Initiated matrices" << std::endl;

    // CPU reference
    double cpu_ms = benchmark_softmax_cpu(M, K);
    double cpu_gflops = ms_to_gflops(M, K, cpu_ms);
    print_results(cpu_ms, cpu_gflops, "CPU");

    // 08: Test simple softmax
    dim3 blockDim_00(BLOCKSIZE_00, BLOCKSIZE_00, 1);
    dim3 gridDim_00(CEIL_DIV(M, BLOCKSIZE_00), CEIL_DIV(K, BLOCKSIZE_00), 1);
    std::cout << "grid: " << gridDim_00.x << std::endl;
    benchmark_softmax_kernel(
        M, K, gridDim_00, blockDim_00,
        [](int M, int K, const float *A, float *C, dim3 gridDim, dim3 blockDim) -> void {
            softmax_simple<<<gridDim, blockDim>>>(M, K, A, C);
        },
        "Kernel 08");

    // 09: Test softmax with partial summation
    benchmark_softmax_kernel(
        M, K, gridDim_00, blockDim_00,
        [](int M, int K, const float *A, float *C, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_sum<<<gridDim, blockDim>>>(M, K, A, C);
        },
        "Kernel 09");

    
    // 09: Test softmax with binary summation

    // Grid for k1 has only one column
    dim3 gridDim_k1(gridDim_00.x, 1, gridDim_00.z);
    benchmark_triple_softmax_kernel(
        M, K, gridDim_00, gridDim_k1, blockDim_00, blockDim_00,
        [](int M, int K, const float *A, float *C, float *temp, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k0<<<gridDim, blockDim>>>(M, K, A, C, temp);
        },
        [](int M, int K, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k1<<<gridDim, blockDim, gridDim_y_k0*BLOCKSIZE_00 * sizeof(float)>>>(M, K, A, C, temp, gridDim_y_k0);
        },
        [](int M, int K, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k2<<<gridDim, blockDim>>>(M, K, A, C, temp, gridDim_y_k0);
        },
        "softmax_binary"
    );

    // 10: Test softmax with binary summation
    dim3 blockDim_10_k0(BLOCKSIZE_00, CEIL_DIV(BLOCKSIZE_00, 2), 1);
    benchmark_triple_softmax_kernel(
        M, K, gridDim_00, gridDim_k1, blockDim_00, blockDim_10_k0,
        [](int M, int K, const float *A, float *C, float *temp, dim3 gridDim, dim3 blockDim) -> void {
            softmax_binary_non_divergent_k0<<<gridDim, blockDim>>>(M, K, A, C, temp);
        },
        [](int M, int K, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_binary_non_divergent_k1<<<gridDim, blockDim, gridDim_y_k0*BLOCKSIZE_00 * sizeof(float)>>>(M, K, A, C, temp, gridDim_y_k0);
        },
        [](int M, int K, const float *A, float *C, float *temp, int gridDim_y_k0, dim3 gridDim, dim3 blockDim) -> void {
            softmax_block_binary_k2<<<gridDim, blockDim>>>(M, K, A, C, temp, gridDim_y_k0);
        },
        "softmax_binary"
    );
}