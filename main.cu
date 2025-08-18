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
    int M_0 = 1027;
    int N_0 = 1023;
    int K_0 = 1025;

    // 00: Test simple kernel
    dim3 blockSimple(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridSimple(CEIL_DIV(M_0, BLOCKSIZE), CEIL_DIV(N_0, BLOCKSIZE), 1);
    run_sgemm_test(
        M_0, N_0, K_0, gridSimple, blockSimple,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block) {
            sgemm_simple<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_simple");

    // 01: Test coalesced kernel
    dim3 blockCoalesced(BLOCKSIZE * BLOCKSIZE);
    dim3 gridCoalesced(CEIL_DIV(N_0, BLOCKSIZE), CEIL_DIV(M_0, BLOCKSIZE), 1);
    run_sgemm_test(
        M_0, N_0, K_0, gridCoalesced, blockCoalesced,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block) {
            sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_coalesced");

    // 02: Test tiled kernel
    dim3 blockTiled(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridTiled(CEIL_DIV(N_0, BLOCKSIZE), CEIL_DIV(M_0, BLOCKSIZE), 1);
    run_sgemm_test(
        M_0, N_0, K_0, gridTiled, blockTiled,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block) {
            sgemm_tiled<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled");

    // 03: Test tiled 2D kernel
    static_assert(BN_03 % TN_03 == 0 && BM_03 % TM_03 == 0,
                  "BN % TN != 0 || BM % TM != 0");
    static_assert(BN_03 / TN_03 == BK_03, "BN / TN != BK");
    static_assert(BM_03 / TM_03 == BK_03, "BM / TM != BK");
    static_assert(BK_03 >= TM_03 && BK_03 >= TN_03, "BK < TM || BK < TN");
    dim3 gridTiled2D(CEIL_DIV(N_0, BN_03), CEIL_DIV(M_0, BM_03), 1);
    dim3 blockTiled2D(BN_03 / TN_03, BM_03 / TM_03, 1);
    run_sgemm_test(
        M_0, N_0, K_0, gridTiled2D, blockTiled2D,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block /*ExtraParams*/) {
            sgemm_tiled_2d<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled_2d");

    // 04: Test tiled 2D kernel with vectorization
    static_assert(BN_04 % TN_04 == 0 && BM_04 % TM_04 == 0,
                  "BN % TN != 0 || BM % TM != 0");
    static_assert(BN_04 / TN_04 == BK_04, "BN / TN != BK");
    static_assert(BM_04 / TM_04 == BK_04, "BM / TM != BK");
    static_assert(BK_04 >= TM_04 && BK_04 >= TN_04, "BK < TM || BK < TN");
    static_assert(BK_04 >= VEC_SIZE_04 && BK_04 % VEC_SIZE_04 == 0,
                  "BK < VEC_SIZE_04 || BK % VEC_SIZE_04 != 0");
    dim3 gridTiled2Dvec(CEIL_DIV(N_0, BN_04), CEIL_DIV(M_0, BM_04), 1);
    dim3 blockTiled2Dvec(BN_04 / TN_04, BM_04 / TM_04, 1);
    run_sgemm_test(
        M_0, N_0, K_0, gridTiled2Dvec, blockTiled2Dvec,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block /*ExtraParams*/) {
            sgemm_tiled_2d_vectorized_1<<<grid, block>>>(M, N, K, alpha, A, B,
                                                         beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled_2d_vectorized_1");

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
    run_sgemm_test(
        M_0, N_0, K_0, gridTiled2Dvec, blockTiled2Dvec,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block /*ExtraParams*/) {
            sgemm_tiled_2d_vectorized_2<<<grid, block>>>(M, N, K, alpha, A, B,
                                                         beta, C);
        },
        1.0f, 0.0f, "sgemm_tiled_2d_vectorized_2");

    // 06: Test warptiling
    static_assert(BN_06 % WN_06 == 0 && BM_06 % WM_06 == 0,
                  "BN % WN != 0 || BM % WM != 0");
    static_assert(BN_06 / TN_06 == BK_06, "BN / TN != BK");
    static_assert(BM_06 / TM_06 == BK_06, "BM / TM != BK");
    static_assert(BK_06 >= TM_06 && BK_06 >= TN_06, "BK < TM || BK < TN");
    static_assert(WN_06 >= TN_06 && WM_06 >= TM_06, "WN < TN || WM < TM");
    static_assert(WN_06 % TN_06 == 0 && WM_06 % TM_06 == 0,
                  "WN % TN != 0 || WM % TM != 0");
    dim3 gridWarptiling(CEIL_DIV(N_0, BN_06), CEIL_DIV(M_0, BM_06), 1);
    dim3 blockWarptiling(BN_06 / TN_06, BM_06 / TM_06, 1);
    run_sgemm_test(
        M_0, N_0, K_0, gridWarptiling, blockWarptiling,
        [](int M, int N, int K, float alpha, const float *A, const float *B,
           float beta, float *C, dim3 grid, dim3 block /*ExtraParams*/) {
            sgemm_warptiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
        },
        1.0f, 0.0f, "sgemm_warptiling");

    return 0;
}
