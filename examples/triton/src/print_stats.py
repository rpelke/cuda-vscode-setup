from triton.runtime.autotuner import Autotuner
import numpy as np
from tabulate import tabulate


def _get_autotuner_obj(kernel):
    fn = kernel
    while hasattr(fn, "fn") and not isinstance(fn, Autotuner):
        fn = fn.fn
    return fn if isinstance(fn, Autotuner) else None


def print_tuning_stats(tuner: Autotuner):
    at = _get_autotuner_obj(tuner)
    if at is None:
        raise Exception("Autotuner not found.")
    if at.cache_results is False:
        print("No tuning results cached.")
    else:
        print("Configs:", len(at.configs))
        print("Tuning keys:", list(at.cache.keys()))

        header = [
            'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'SWIZZLE_M', 'num_warps', 'num_ctas',
            'num_stages', 'maxnreg', 'pre_hook', 'ir_override', 'Ø time_ms'
        ]
        data = []

        for cfg in at.configs:
            if cfg in at.configs_timings:
                data.append([
                    cfg.kwargs['BLOCK_SIZE_M'], cfg.kwargs['BLOCK_SIZE_N'],
                    cfg.kwargs['BLOCK_SIZE_K'], cfg.kwargs['SWIZZLE_M'], cfg.num_warps,
                    cfg.num_ctas, cfg.num_stages,
                    str(cfg.maxnreg),
                    str(cfg.pre_hook),
                    str(cfg.ir_override), f"{float(np.mean(at.configs_timings[cfg])):8.3f}"
                ])

        print(tabulate(data, headers=header, tablefmt="rounded_grid"))
