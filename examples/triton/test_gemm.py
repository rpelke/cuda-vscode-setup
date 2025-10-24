from src.gemm_kernel import check_and_launch_matmul, matmul_kernel
from src.print_stats import print_tuning_stats
from src.torch_kernel import time_gpu
import triton
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Test
torch.manual_seed(0)
a = (torch.rand((1280, 1280), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
b = (torch.rand((1280, 1280), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
triton_output = check_and_launch_matmul(a, b)
torch_output = torch.matmul(a, b)

# Compare outputs and benchmark
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
    print("Test passed! ✅")
    print_tuning_stats(matmul_kernel)

    torch_ms = time_gpu(lambda: torch.matmul(a, b), warmup=5, iters=20)
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: ( \
        triton.cdiv(M, META['BLOCK_SIZE_M']) * \
        triton.cdiv(N, META['BLOCK_SIZE_N']), )
    tri_ms = time_gpu(lambda: matmul_kernel[grid](a, b, c, M, N, K), warmup=5, iters=20)

    print(f"Torch time: {torch_ms:.3f} ms")
    print(f"Triton best time: {tri_ms:.3f} ms")
    print(f"Speedup: {torch_ms / tri_ms:.2f}x")
else:
    print("Test failed! ❌")
