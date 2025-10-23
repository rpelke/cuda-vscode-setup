#ifndef BENCHMARK_SGEMM_CUH
#define BENCHMARK_SGEMM_BENCHMARK_CUH

#include "benchmark.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <functional>
#include <string>

typedef std::function<void(int, int, int, float, const float *, const float *,
                           float, float *, dim3, dim3)>
    sgemm_kernel_t;

class SGEMMBenchmark : public Benchmark {
  public:
    SGEMMBenchmark();
    SGEMMBenchmark(const SGEMMBenchmark &) = delete;
    virtual ~SGEMMBenchmark() { free_device_mem(d_A, d_B, d_C, d_C_init_helper); }

    void start_benchmarks(int M, int K, int N, float alpha, float beta);

    void start_softmax_benchmarks(int M, int K);

  private:
    double benchmark_cpu(int M, int K, int N, float alpha, float beta);
    double benchmark_cublas_sgemv(int M, int N, float alpha, float beta);
    void benchmark_kernel(int M, int K, int N, float alpha, float beta,
                          dim3 gridDim, dim3 blockDim, sgemm_kernel_t launcher,
                          std::string kernel_name, float atol);
    double ms_to_gflops(int M, int K, int N, double ms);

    std::vector<float> h_A, h_B, h_C, h_C_init, h_C_cpu, h_C_cublas;
    float *d_A, *d_B, *d_C, *d_C_init_helper;
};

#endif // BENCHMARK_SGEMM_CUH