from src.gemm_kernel_0 import matmul_kernel
from src.print_stats import get_autotuner_obj
from src.torch_kernel import time_gpu
from src.generic import check_and_launch_matmul
import src.gemm_kernel_0 as k0
import src.gemm_kernel_1 as k1
from tabulate import tabulate
import torch
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

kernels = [k0, k1]


def create_grids(M, N):
    return {
        k0:
        lambda META:
        (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), ),
        k1:
        lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
                      META["SPLIT_K"])
    }


def gflops(m, n, k, ms):
    return (2.0 * m * n * k) / (ms * 1e-3) / 1e9


def rand_mat(m, k, dtype=torch.float32):
    return (torch.rand((m, k), device=DEVICE, dtype=dtype) - 0.5).contiguous()


def bench_square(min_sz=256, max_sz=2048, step=128, dtype=torch.float32, warmup=5, iters=20):
    header_times = [
        "M", "N", "K", "Triton ms", "Torch ms", "Triton GF/s", "Torch GF/s", "Speedup", "OK"
    ]
    header_best_configs = {}
    data_best_configs = {}

    data_times = {}
    for kernel in kernels:
        data_times[kernel] = []
        data_best_configs[kernel] = []

    for s in range(min_sz, max_sz + 1, step):
        M = N = K = s
        a = rand_mat(M, K, dtype)
        b = rand_mat(K, N, dtype)
        c_ref = torch.matmul(a, b)

        grids = create_grids(M, N)

        for kernel in kernels:
            at = get_autotuner_obj(kernel.matmul_kernel)
            c_tri = check_and_launch_matmul(a, b, kernel=kernel.matmul_kernel, grid=grids[kernel])
            ok = torch.allclose(c_tri, c_ref, atol=1e-4, rtol=1e-5)

            c = torch.empty((M, N), device=a.device, dtype=torch.float32)

            tri_ms = time_gpu(lambda: kernel.matmul_kernel[grids[kernel]](a, b, c, M, N, K),
                              warmup=warmup,
                              iters=iters)
            torch_ms = time_gpu(lambda: torch.matmul(a, b), warmup=warmup, iters=iters)

            tri_gf = gflops(M, N, K, tri_ms)
            torch_gf = gflops(M, N, K, torch_ms)
            speedup = torch_ms / tri_ms if tri_ms > 0 else float('nan')

            at = get_autotuner_obj(kernel.matmul_kernel)
            header_best_configs[kernel] = [
                'M', 'N', 'K', *at.best_config.kwargs.keys(), 'num_warps', 'num_ctas', 'num_stages',
                'maxnreg', 'pre_hook', 'ir_override', 'Ø time_ms'
            ]
            data_best_configs[kernel].append([
                M, N, K, *at.best_config.kwargs.values(), at.best_config.num_warps,
                at.best_config.num_ctas, at.best_config.num_stages,
                str(at.best_config.maxnreg),
                str(at.best_config.pre_hook),
                str(at.best_config.ir_override), f"{tri_ms:8.3f}"
            ])

            data_times[kernel].append([
                M, N, K, f"{tri_ms:9.3f}", f"{torch_ms:9.3f}", f"{tri_gf:10.1f}",
                f"{torch_gf:11.1f}", f"{speedup:8.3f}", '✅' if ok else '❌'
            ])

    for kernel in kernels:
        print(f"\n=== Benchmark results for kernel: {kernel.__name__} ===")
        print("GEMM: Triton vs. Torch:")
        print(tabulate(data_times[kernel], headers=header_times, tablefmt="rounded_grid"))
        print("\nBest Tuning Configurations:")
        print(
            tabulate(data_best_configs[kernel],
                     headers=header_best_configs[kernel],
                     tablefmt="rounded_grid"))


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "GPU required"
    bench_square(min_sz=256, max_sz=2048, step=128, dtype=torch.float32, warmup=5, iters=20)
