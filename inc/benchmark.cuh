#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>
#include <string>

typedef std::function<void(int, int, int, float, const float *, const float *,
                           float, float *, dim3, dim3)>
    sgemm_kernel_t;

typedef std::function<void(int, int, const float *, float *, dim3, dim3)>
    softmax_kernel_t;

typedef std::function<void(int, int, const float *, float *, float *, dim3,
                           dim3)>
    softmax_init_kernel_t;

typedef std::function<void(int, int, const float *, float *, float *, int, dim3,
                           dim3)>
    softmax_followUp_kernel_t;

class Benchmark {
  public:
    Benchmark();
    Benchmark(const Benchmark &) = delete;
    virtual ~Benchmark() {}

    double
    benchmark_cublas(std::function<cublasStatus_t(cublasHandle_t)> cublasFunc,
                     std::function<void(cudaStream_t)> resetResultTensor,
                     int warmup = 10, int iters = 10);

    // GEMM-like init
    static void copy_to_device(float *&d_A, float *&d_B, float *&d_C,
                               float *&d_C_init_helper, std::vector<float> &h_A,
                               std::vector<float> &h_B, std::vector<float> &h_C,
                               std::vector<float> &res_vector, int M, int K,
                               int N);
    // GEMV-like init (B 1-dim)
    static void copy_to_device(float *&d_A, float *&d_B, float *&d_C,
                               float *&d_C_init_helper, std::vector<float> &h_A,
                               std::vector<float> &h_B, std::vector<float> &h_C,
                               std::vector<float> &res_vector, int M, int N);
    // softmax-like init
    static void copy_to_device(float *&d_A, float *&d_C,
                               float *&d_C_init_helper, std::vector<float> &h_A,
                               std::vector<float> &h_C,
                               std::vector<float> &res_vector, int M, int N);

    // GEMM-like init
    void init_matrices(std::vector<float> &h_A, std::vector<float> &h_B,
                       std::vector<float> &h_C, std::vector<float> &h_C_init,
                       int M, int K, int N);
    void init_matrices(std::vector<float> &h_A, std::vector<float> &h_B,
                       std::vector<float> &h_C, std::vector<float> &h_C_init,
                       std::vector<float> &h_C_cpu,
                       std::vector<float> &h_C_cublas, int M, int K, int N);
    // GEMV-like init
    void init_matrices(std::vector<float> &h_A, std::vector<float> &h_B,
                       std::vector<float> &h_C, std::vector<float> &h_C_init,
                       int M, int N);
    // softmax-like init
    void init_matrices(std::vector<float> &h_A, std::vector<float> &h_C,
                       std::vector<float> &h_C_init, int M, int N);
    void init_matrices(std::vector<float> &h_A, std::vector<float> &h_C,
                       std::vector<float> &h_C_init,
                       std::vector<float> &h_C_cpu,
                       std::vector<float> &h_C_cublas, int M, int N);

  protected:
    void print_results(double ms, double gflops, std::string name);
    void copy_results_to_host(float *&d_C, int M, int N,
                              std::vector<float> &res_vector);
    void free_device_mem(float *d_A, float *d_B, float *d_C,
                         float *d_C_init_helper);
    void free_device_mem(float *d_A, float *d_C, float *d_C_init_helper);
    bool validate_results(std::vector<float> &C_test,
                          std::vector<float> &h_C_cpu, std::string test_name,
                          int M, int N, float atol);
};

#endif // BENCHMARK_CUH
