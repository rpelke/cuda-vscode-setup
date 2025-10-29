import src.gemm_kernel_0 as k0
import src.gemm_kernel_1 as k1
from src.generic import check_and_launch_matmul
from src.print_stats import print_tuning_stats
from src.torch_kernel import time_gpu
import triton
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

M = N = K = 1280

kernels = [k0, k1]

grids = {
    k0:
    lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), ),
    k1:
    lambda META:
    (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META["SPLIT_K"])
}

torch.manual_seed(0)
a = (torch.rand((M, K), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
b = (torch.rand((K, N), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
torch_output = torch.matmul(a, b)
torch_ms = time_gpu(lambda: torch.matmul(a, b), warmup=5, iters=20)

for kernel in kernels:
    if kernel not in grids:
        raise ValueError(f"No grid defined for kernel {kernel}")

    triton_output = check_and_launch_matmul(a, b, kernel=kernel.matmul_kernel, grid=grids[kernel])

    # Compare outputs and benchmark
    if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
        print("Test passed! ✅")
        print_tuning_stats(kernel.matmul_kernel)

        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
        tri_ms = time_gpu(lambda: kernel.matmul_kernel[grids[kernel]](a, b, c, M, N, K),
                          warmup=5,
                          iters=20)

        print(f"Torch time: {torch_ms:.3f} ms")
        print(f"Triton best time: {tri_ms:.3f} ms")
        print(f"Speedup: {torch_ms / tri_ms:.2f}x")
    else:
        print("Test failed! ❌")
