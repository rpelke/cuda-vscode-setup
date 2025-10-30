import math
import torch
import triton
import triton.language as tl
from itertools import product


def get_cuda_autotune_config():
    BM = [64, 128]
    BN = [64, 128]
    BK = [16, 32]
    NUM_WARPS = [4, 8]
    NUM_STAGES = [3, 4]
    SWIZZLE_M = [4, 6]
    NUM_SMS = [torch.cuda.get_device_properties("cuda").multi_processor_count]

    specs = [
        (bm, bn, bk, w, st, sw, num_sm)
        for bm, bn, w, st, sw, num_sm in product(BM, BN, NUM_WARPS, NUM_STAGES, SWIZZLE_M, NUM_SMS)
        for bk in BK if bk <= bm and bk <= bn
    ]
    return [
        triton.Config(kwargs={
            'BLOCK_SIZE_M': bm,
            'BLOCK_SIZE_N': bn,
            'BLOCK_SIZE_K': bk,
            'SWIZZLE_M': sw,
            'NUM_SMS': num_sm
        },
                      num_warps=w,
                      num_stages=st) for (bm, bn, bk, w, st, sw, num_sm) in specs
    ]


def validate_results(*args, **kwargs):
    a = args[0]['a_ptr']
    b = args[0]['b_ptr']
    c = args[0]['c_ptr']
    torch.cuda.synchronize(c.device)
    triton_output = c
    torch_output = torch.matmul(a, b)
    if not torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
        print("Post hook: Test failed! ❌")


# Filter configurations depending on the matrix sizes
def _prune_configs(configs, named_args, **kwargs):
    M, N, K = named_args['M'], named_args['N'], named_args['K']
    pruned_configs = [
        c for c in configs if \
            (c.kwargs['BLOCK_SIZE_M']<=M) and \
            (c.kwargs['BLOCK_SIZE_N']<=N) and \
            (c.kwargs['BLOCK_SIZE_K']<=K) and \
            (c.kwargs['SWIZZLE_M'] <= math.ceil(M / c.kwargs['BLOCK_SIZE_M']))
    ]
    if len(pruned_configs) == 0:
        raise Exception("No valid configurations found for the given matrix size.")
    return pruned_configs


def _zero_output_tuning(*args, **kwargs):
    args[0]['c_ptr'].zero_()


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(configs=get_cuda_autotune_config(),
                 key=['M', 'N', 'K'],
                 prune_configs_by={"early_config_prune": _prune_configs},
                 post_hook=validate_results,
                 pre_hook=_zero_output_tuning,
                 cache_results=True)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SWIZZLE_M: tl.constexpr,
                  NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = SWIZZLE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, SWIZZLE_M, NUM_SMS)

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
            b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        tl.store(c_ptrs, accumulator, mask=c_mask)
