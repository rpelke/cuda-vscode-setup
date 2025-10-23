#include "sgemm/benchmark_sgemm.cuh"

int main() {
    int M_0 = 1027;
    int N_0 = 1023;
    int K_0 = 1025;

    SGEMMBenchmark sb;
    sb.start_benchmarks(M_0, K_0, N_0, 2.0f, 3.0f);
}
