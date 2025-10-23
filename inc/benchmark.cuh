#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <cuda_runtime.h>
#include <functional>
#include <string>

typedef std::function<void(int, int, int, float, const float *, const float *,
                           float, float *, dim3, dim3)>
    sgemm_kernel_t;

typedef std::function<void(int, int, const float *,
                           float *, dim3, dim3)>
    softmax_kernel_t;

typedef std::function<void(int, int, const float *,
                           float *, float *, dim3, dim3)>
    softmax_init_kernel_t;

typedef std::function<void(int, int, const float *,
                           float *, float *, int, dim3, dim3)>
    softmax_followUp_kernel_t;

class Benchmark {
  public:
    Benchmark();
    Benchmark(const Benchmark &) = delete;
    virtual ~Benchmark() { free_device_mem(); }

    void start_benchmarks(int M, int K, int N, float alpha, float beta);

    void start_softmax_benchmarks(int M, int K);

  private:
    void init_matrices(int M, int K, int N);
    void init_matrices(int M, int K);
    void copy_to_device(int M, int K, int N, std::vector<float> &res_vector);
    void copy_to_device(int M, int K, std::vector<float> &res_vector);
    void copy_results_to_host(int M, int N, std::vector<float> &res_vector);
    void free_device_mem();
    bool validate_results(std::vector<float> &C_test, std::string test_name,
                          int M, int N, float atol);
    double benchmark_cpu(int M, int K, int N, float alpha, float beta);
    double benchmark_softmax_cpu(int M, int K);
    double benchmark_cublas(int M, int K, int N, float alpha, float beta);
    void benchmark_kernel(int M, int K, int N, float alpha, float beta,
                          dim3 gridDim, dim3 blockDim, sgemm_kernel_t launcher,
                          std::string kernel_name, float atol);
    void benchmark_softmax_kernel(int M, int K,
                          dim3 gridDim, dim3 blockDim, softmax_kernel_t launcher,
                          std::string kernel_name, float atol);
    void benchmark_triple_softmax_kernel(int M, int K,
                          dim3 gridDim_k0, dim3 gridDim_k1, dim3 gridDim_k2, dim3 blockDim, dim3 blockDim_k0, softmax_init_kernel_t k0, softmax_followUp_kernel_t k1, softmax_followUp_kernel_t k2, std::string kernel_name, float atol);
    void benchmark_recursive_softmax_kernel(int M, int K, float atol);
    double ms_to_gflops(int M, int K, int N, double ms);
    double ms_to_gflops(int M, int K, double ms);

    std::vector<float> h_A, h_B, h_C, h_C_init, h_C_cpu, h_C_cublas;
    float *d_A, *d_B, *d_C, *d_C_init_helper;
};

#endif // BENCHMARK_CUH