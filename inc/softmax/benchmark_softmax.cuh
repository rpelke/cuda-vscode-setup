#ifndef BENCHMARK_SOFTMAX_CUH
#define BENCHMARK_SOFTMAX_BENCHMARK_CUH

#include "benchmark.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>
#include <string>

typedef std::function<void(int, int, const float *, float *, dim3, dim3)>
    softmax_kernel_t;

typedef std::function<void(int, int, const float *, float *, float *, dim3,
                           dim3)>
    softmax_init_kernel_t;

typedef std::function<void(int, int, const float *, float *, float *, int, dim3,
                           dim3)>
    softmax_followUp_kernel_t;

class SoftmaxBenchmark : Benchmark {
  public:
    SoftmaxBenchmark();
    SoftmaxBenchmark(const SoftmaxBenchmark &) = delete;
    virtual ~SoftmaxBenchmark() { free_device_mem(d_A, d_C, d_C_init_helper); }

    void start_benchmarks(int M, int K);

  private:
    double benchmark_cpu(int M, int K);
    void benchmark_kernel(int M, int K, dim3 gridDim, dim3 blockDim,
                          softmax_kernel_t launcher, std::string kernel_name,
                          float atol);
    void benchmark_triple_softmax_kernel(int M, int K, dim3 gridDim_k0,
                                         dim3 gridDim_k1, dim3 gridDim_k2,
                                         dim3 blockDim, dim3 blockDim_k0,
                                         softmax_init_kernel_t k0,
                                         softmax_followUp_kernel_t k1,
                                         softmax_followUp_kernel_t k2,
                                         std::string kernel_name, float atol);
    void benchmark_recursive_softmax_kernel(int M, int K, float atol);
    double ms_to_gflops(int M, int K, double ms);

    std::vector<float> h_A, h_C, h_C_init, h_C_cpu, h_C_cublas;
    float *d_A, *d_B, *d_C, *d_C_init_helper;
};

#endif // BENCHMARK_SOFTMAX_CUH
