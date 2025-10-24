#include "softmax/softmax.cuh"
#include <cmath>
#include <iostream>

void cpu_softmax(int M, int N, const std::vector<float> &A,
                 std::vector<float> &C) {

    float sums[M] = {0.0f};

    // Collect sums
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            sums[i] += exp(A[i * N + j]);
        }
    }

    // Calculate results
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = exp(C[i * N + j]) / sums[i];
        }
    }
}
