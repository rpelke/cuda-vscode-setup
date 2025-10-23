#include "sgemm/sgemm.cuh"

void cpu_sgemm(int M, int N, int K, float alpha, const std::vector<float> &A,
               const std::vector<float> &B, float beta, std::vector<float> &C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k)
                tmp += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * tmp + beta * C[i * N + j];
        }
    }
}
