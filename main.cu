#include "benchmark.cuh"

int main() {
    // TODO: Boundary checks for kernel 07 (1027, 1023, 1025)
    int M_0 = 1024;
    int N_0 = 1024;
    int K_0 = 1024;

    Benchmark b;
    b.start_benchmarks(M_0, K_0, N_0, 1.0f, 0.0f);
}
