#include "softmax/benchmark_softmax.cuh"

int main() {
    int M_0 = 1027;
    int N_0 = 1025;

    SoftmaxBenchmark sb;
    sb.start_benchmarks(M_0, N_0);
}
