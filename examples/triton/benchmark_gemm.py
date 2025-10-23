from src.gemm_kernel import check_and_launch_matmul, matmul_kernel
import torch
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def gflops(m, n, k, ms):
    return (2.0 * m * n * k) / (ms * 1e-3) / 1e9


@torch.inference_mode()
def time_gpu(fn, warmup=5, iters=20):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    n = len(times_ms)
    if n % 2 == 1:
        median_ms = times_ms[n // 2]
    else:
        median_ms = 0.5 * (times_ms[n // 2 - 1] + times_ms[n // 2])
    return median_ms


def rand_mat(m, k, dtype=torch.float32):
    return (torch.rand((m, k), device=DEVICE, dtype=dtype) - 0.5).contiguous()


def bench_square(min_sz=128, max_sz=2048, step=128, dtype=torch.float32, warmup=5, iters=20):
    print(
        f"{'M':>5} {'N':>5} {'K':>5} | {'Tri ms':>9} {'Torch ms':>9} | {'Tri GF/s':>10} {'Torch GF/s':>11} | {'Speedup':>8} {'OK':>3}"
    )
    print("-" * 89)
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

        print(
            f"{M:5d} {N:5d} {K:5d} | {tri_ms:9.3f} {torch_ms:9.3f} | {tri_gf:10.1f} {torch_gf:11.1f} | {speedup:8.3f} {'✅' if ok else '❌':>3}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "GPU required"
    bench_square(min_sz=128, max_sz=2048, step=128, dtype=torch.float32, warmup=5, iters=20)
