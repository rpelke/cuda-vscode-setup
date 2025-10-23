#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

#define CUDA_CHECK(val) cudaCheck((val), __FILE__, __LINE__)
inline void cudaCheck(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " -> "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#endif // UTILS_CUH