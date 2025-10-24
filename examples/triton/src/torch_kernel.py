import torch


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
