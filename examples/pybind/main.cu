#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#include "sgemm.cuh"

torch::Tensor matmul_linear(torch::Tensor x, torch::Tensor W) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, in_features]");
    TORCH_CHECK(W.dim() == 2, "W must be 2D [out_features, in_features]");
    TORCH_CHECK(x.size(1) == W.size(1),
                "in_features mismatch: x.size(1) != W.size(1)");
    TORCH_CHECK(x.dtype() == W.dtype(), "x and W must have same dtype");
    TORCH_CHECK(x.device() == W.device(), "x and W must be on the same device");
    return at::mm(x.contiguous(), W.t().contiguous());
}

torch::Tensor matmul_linear_cuda(torch::Tensor x, torch::Tensor W) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda(), "x and W must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && W.dtype() == torch::kFloat32,
                "x and W must be float32 for this kernel");
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2, "x [M,K], W [N_out,K]");
    TORCH_CHECK(x.size(1) == W.size(1),
                "in_features mismatch: x.size(1) != W.size(1)");
    TORCH_CHECK(x.device() == W.device(), "x and W must be on same device");

    // Shapes
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = W.size(0);

    auto B = W.t().contiguous();
    auto A = x.contiguous();
    auto C = torch::empty({M, N}, x.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float *A_ptr = A.data_ptr<float>(); // M×K
    const float *B_ptr = B.data_ptr<float>(); // K×N
    float *C_ptr = C.data_ptr<float>();       // M×N

    static_assert(TN_07 * TM_07 * 32 == BLOCKSIZE_07 * BLOCKSIZE_07);
    dim3 gridDim_07(CEIL_DIV(N, BLOCKSIZE_07), CEIL_DIV(M, BLOCKSIZE_07), 1);
    dim3 blockDim_07(BLOCKSIZE_07 / TN_07, BLOCKSIZE_07 / TM_07, 1);

    // Use PyTorch stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    std::cout << "Launching kernel..." << std::endl;
    sgemm_tensorcores<<<gridDim_07, blockDim_07, 0, stream>>>(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), alpha,
        A_ptr, B_ptr, beta, C_ptr);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "sgemm_tensorcores launch failed");

    return C;
}

PYBIND11_MODULE(mymatmul, m) {
    m.def("matmul_linear", &matmul_linear, "Linear without bias (y = x @ W^T)");
    m.def("matmul_linear_cuda", &matmul_linear_cuda,
          "Linear without bias (y = x @ W^T) [CUDA]");
}
