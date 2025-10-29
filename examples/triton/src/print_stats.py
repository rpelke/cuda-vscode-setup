from triton.runtime.autotuner import Autotuner
import numpy as np
from tabulate import tabulate


def get_autotuner_obj(tuner: Autotuner):
    fn = tuner
    while hasattr(fn, "fn") and not isinstance(fn, Autotuner):
        fn = fn.fn
    return fn if isinstance(fn, Autotuner) else None


def print_tuning_stats(tuner: Autotuner):
    at = get_autotuner_obj(tuner)
    if at is None:
        raise Exception("Autotuner not found.")
    if at.cache_results is False:
        print("No tuning results cached.")
    else:
        print("Configs:", len(at.configs))
        print("Tuning keys:", list(at.cache.keys()))

        header = [
            *at.best_config.kwargs.keys(), 'num_warps', 'num_ctas', 'num_stages', 'maxnreg',
            'pre_hook', 'ir_override', 'Ø time_ms'
        ]
        data = []

        sorted_timings = sorted(at.configs_timings.items(), key=lambda x: np.mean(x[1]))

        for cfg, times in sorted_timings:
            data.append([
                *cfg.kwargs.values(), cfg.num_warps, cfg.num_ctas, cfg.num_stages,
                str(cfg.maxnreg),
                str(cfg.pre_hook),
                str(cfg.ir_override), f"{float(np.mean(times)):8.3f}"
            ])

        print(tabulate(data, headers=header, tablefmt="rounded_grid"))
