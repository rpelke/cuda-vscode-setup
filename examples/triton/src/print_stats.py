from triton.runtime.autotuner import Autotuner


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
        print("--- Best config per key ---")
        for key, cfg in at.cache.items():
            print(key, cfg.__dict__)
        print("--- Timings per config ---")
        for cfg, t in at.configs_timings.items():
            print(cfg.__dict__, "->", t, "ms")
