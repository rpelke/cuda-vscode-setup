#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "sgemm.cuh"

// run_sgemm_test now takes grid and block dimensions as parameters
template <typename KernelLauncher, typename... ExtraParams>
void run_sgemm_test(int M, int N, int K, dim3 gridDim, dim3 blockDim,
                    KernelLauncher launcher, float alpha = 1.0f,
                    float beta = 0.0f, const std::string &name = "SGEMM",
                    ExtraParams &&...extraParams) {
    std::cout << "=== Testing '" << name << "' M=" << M << ", N=" << N
              << ", K=" << K << " ===\n";

    // Allocate and initialize host data
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);
    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C_ref[i] = h_C[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy host -> device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the specified kernel
    launcher(M, N, K, alpha, d_A, d_B, beta, d_C, gridDim, blockDim,
             std::forward<ExtraParams>(extraParams)...);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy device -> host
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
                  << "\n";
    }

    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C_ref);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // CPU and GPU performance
    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    double gpu_gflops = 2.0 * M * N * K / (gpu_ms * 1e6);
    double cpu_gflops = 2.0 * M * N * K / (cpu_ms * 1e6);
    std::cout << "GPU   : " << gpu_ms << " ms, " << gpu_gflops << " GFLOPS\n";
    std::cout << "CPU   : " << cpu_ms << " ms, " << cpu_gflops << " GFLOPS\n";

    // Validation
    int mismatches = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float e = std::abs(h_C_ref[i] - h_C[i]);
        if (e > 1e-3f) {
            ++mismatches;
            max_err = std::max(max_err, e);
        }
    }
    if (mismatches == 0) {
        std::cout << " TEST PASSED\n\n";
    } else {
        std::cout << " TEST FAILED: " << mismatches
                  << " mismatches, max error = " << max_err << "\n\n";
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 1023;
    int N = 1025;
    int K = 1029;

    // Test simple kernel
    dim3 blockSimple(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridSimple(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1);
    run_sgemm_test(
        M, N, K, gridSimple, blockSimple,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block) {
            sgemm_simple<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_simple");

    // Test coalesced kernel
    dim3 blockCoalesced(BLOCKSIZE * BLOCKSIZE);
    dim3 gridCoalesced(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
    run_sgemm_test(
        M, N, K, gridCoalesced, blockCoalesced,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block) {
            sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_coalesced");

    // Test tiled kernel
    dim3 blockTiled(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridTiled(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
    run_sgemm_test(
        M, N, K, gridTiled, blockTiled,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block) {
            sgemm_tiled<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled");

    // Test tiled 2D kernel
    static_assert(BN % TN == 0 && BM % TM == 0, "BN % TN != 0 || BM % TM != 0");
    static_assert(BN / TN == BK, "BN / TN != BK");
    static_assert(BM / TM == BK, "BM / TM != BK");
    static_assert(BK >= TM && BK >= TN, "BK < TM || BK < TN");
    dim3 gridTiled2D(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    dim3 blockTiled2D(BN / TN, BM / TM, 1);
    run_sgemm_test(
        M, N, K, gridTiled2D, blockTiled2D,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block /*ExtraParams*/) {
            sgemm_tiled_2d<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled_2d");

    // Test tiled 2D kernel with vectorization
    static_assert(BN % TN == 0 && BM % TM == 0, "BN % TN != 0 || BM % TM != 0");
    static_assert(BN / TN == BK, "BN / TN != BK");
    static_assert(BM / TM == BK, "BM / TM != BK");
    static_assert(BK >= TM && BK >= TN, "BK < TM || BK < TN");
    static_assert(BK >= VEC_SIZE && BK % VEC_SIZE == 0,
                  "BK < VEC_SIZE || BK % VEC_SIZE != 0");
    dim3 gridTiled2Dvec(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    dim3 blockTiled2Dvec(BN / TN, BM / TM, 1);
    run_sgemm_test(
        M, N, K, gridTiled2Dvec, blockTiled2Dvec,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block /*ExtraParams*/) {
            sgemm_tiled_2d_vectorized_1<<<grid, block>>>(M, N, K, alpha, A, B,
                                                         beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled_2d_vectorized_1");

    // Test tiled 2D kernel with vectorization
    static_assert(BN % TN == 0 && BM % TM == 0, "BN % TN != 0 || BM % TM != 0");
    static_assert(BN / TN == BK, "BN / TN != BK");
    static_assert(BM / TM == BK, "BM / TM != BK");
    static_assert(BK >= TM && BK >= TN, "BK < TM || BK < TN");
    static_assert(BK >= VEC_SIZE && BK % VEC_SIZE == 0,
                  "BK < VEC_SIZE || BK % VEC_SIZE != 0");
    static_assert((TN >= VEC_SIZE) && (TN % VEC_SIZE) == 0,
                  "TN < VEC_SIZE || TN % VEC_SIZE != 0");
    run_sgemm_test(
        M, N, K, gridTiled2Dvec, blockTiled2Dvec,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block /*ExtraParams*/) {
            sgemm_tiled_2d_vectorized_2<<<grid, block>>>(M, N, K, alpha, A, B,
                                                         beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled_2d_vectorized_2");

    return 0;
}
