from src.gemm_kernel import check_and_launch_matmul, matmul_kernel
from src.print_stats import get_autotuner_obj
from src.torch_kernel import time_gpu
from tabulate import tabulate
import torch
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def gflops(m, n, k, ms):
    return (2.0 * m * n * k) / (ms * 1e-3) / 1e9


def rand_mat(m, k, dtype=torch.float32):
    return (torch.rand((m, k), device=DEVICE, dtype=dtype) - 0.5).contiguous()


def bench_square(min_sz=256, max_sz=2048, step=128, dtype=torch.float32, warmup=5, iters=20):
    header_times = [
        "M", "N", "K", "Triton ms", "Torch ms", "Triton GF/s", "Torch GF/s", "Speedup", "OK"
    ]
    header_best_configs = [
        'M', 'N', 'K', 'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'SWIZZLE_M', 'num_warps',
        'num_ctas', 'num_stages', 'maxnreg', 'pre_hook', 'ir_override', 'Ø time_ms'
    ]
    data_times = []
    data_best_configs = []
    for s in range(min_sz, max_sz + 1, step):
        M = N = K = s
        a = rand_mat(M, K, dtype)
        b = rand_mat(K, N, dtype)

        c_tri = check_and_launch_matmul(a, b)
        c_ref = torch.matmul(a, b)
        ok = torch.allclose(c_tri, c_ref, atol=1e-4, rtol=1e-5)

        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        grid = lambda META: ( \
            triton.cdiv(M, META['BLOCK_SIZE_M']) * \
            triton.cdiv(N, META['BLOCK_SIZE_N']), )

        tri_ms = time_gpu(lambda: matmul_kernel[grid](a, b, c, M, N, K), warmup=warmup, iters=iters)
        torch_ms = time_gpu(lambda: torch.matmul(a, b), warmup=warmup, iters=iters)

        tri_gf = gflops(M, N, K, tri_ms)
        torch_gf = gflops(M, N, K, torch_ms)
        speedup = torch_ms / tri_ms if tri_ms > 0 else float('nan')

        at = get_autotuner_obj(matmul_kernel)
        data_best_configs.append([
            M, N, K, at.best_config.kwargs['BLOCK_SIZE_M'], at.best_config.kwargs['BLOCK_SIZE_N'],
            at.best_config.kwargs['BLOCK_SIZE_K'], at.best_config.kwargs['SWIZZLE_M'],
            at.best_config.num_warps, at.best_config.num_ctas, at.best_config.num_stages,
            str(at.best_config.maxnreg),
            str(at.best_config.pre_hook),
            str(at.best_config.ir_override), f"{tri_ms:8.3f}"
        ])

        data_times.append([
            M, N, K, f"{tri_ms:9.3f}", f"{torch_ms:9.3f}", f"{tri_gf:10.1f}", f"{torch_gf:11.1f}",
            f"{speedup:8.3f}", '✅' if ok else '❌'
        ])
    print("\n=== GEMM: Triton vs. Torch ===")
    print(tabulate(data_times, headers=header_times, tablefmt="rounded_grid"))
    print("\n=== Best Tuning Configurations ===")
    print(tabulate(data_best_configs, headers=header_best_configs, tablefmt="rounded_grid"))


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "GPU required"
    bench_square(min_sz=256, max_sz=2048, step=128, dtype=torch.float32, warmup=5, iters=20)
