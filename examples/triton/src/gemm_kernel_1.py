import math
import torch
import triton
import triton.language as tl
from itertools import product
from .generic import pid_mn_from_pid


def get_cuda_autotune_config():
    BM = [64, 128]
    BN = [64, 128]
    BK = [16, 32]
    NUM_WARPS = [4, 8]
    NUM_STAGES = [3, 4]
    SWIZZLE_M = [4, 6]
    SPLIT_K = [2, 3]

    specs = [(bm, bn, bk, w, st, sw, sk)
             for bm, bn, w, st, sw, sk in product(BM, BN, NUM_WARPS, NUM_STAGES, SWIZZLE_M, SPLIT_K)
             for bk in BK if bk <= bm and bk <= bn]
    return [
        triton.Config(kwargs={
            'BLOCK_SIZE_M': bm,
            'BLOCK_SIZE_N': bn,
            'BLOCK_SIZE_K': bk,
            'SWIZZLE_M': sw,
            'SPLIT_K': sk
        },
                      num_warps=w,
                      num_stages=st) for (bm, bn, bk, w, st, sw, sk) in specs
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


@triton.autotune(configs=get_cuda_autotune_config(),
                 key=['M', 'N', 'K'],
                 prune_configs_by={"early_config_prune": _prune_configs},
                 post_hook=validate_results,
                 pre_hook=_zero_output_tuning,
                 cache_results=True)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SWIZZLE_M: tl.constexpr,
                  SPLIT_K: tl.constexpr):
    """
    C = A x B
    A: (M, K) float32
    B: (K, N) float32
    C: (M, N) float32
    """
    C = c_ptr

    # Block index conversion
    pid_m, pid_n = pid_mn_from_pid(BLOCK_SIZE_M, BLOCK_SIZE_N, M, N, SWIZZLE_M)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

    A_tile_ptr = a_ptr + (rm[:, None] * K + rk[None, :])
    B_tile_ptr = b_ptr + (rk[:, None] * N + rn[None, :])
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        a = tl.load(A_tile_ptr, mask=rk[None, :] < k_remaining, other=0.0)
        b = tl.load(B_tile_ptr, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A_tile_ptr += BLOCK_SIZE_K * SPLIT_K
        B_tile_ptr += BLOCK_SIZE_K * SPLIT_K * N

    acc = acc.to(C.dtype.element_ty)
    C_tile_ptr = c_ptr + (rm[:, None] * N + rn[None, :])
    tl.atomic_add(C_tile_ptr, acc, sem="relaxed")
