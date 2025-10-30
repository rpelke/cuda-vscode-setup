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

    specs = [(bm, bn, bk, w, st, sw)
             for bm, bn, w, st, sw in product(BM, BN, NUM_WARPS, NUM_STAGES, SWIZZLE_M) for bk in BK
             if bk <= bm and bk <= bn]
    return [
        triton.Config(kwargs={
            'BLOCK_SIZE_M': bm,
            'BLOCK_SIZE_N': bn,
            'BLOCK_SIZE_K': bk,
            'SWIZZLE_M': sw
        },
                      num_warps=w,
                      num_stages=st) for (bm, bn, bk, w, st, sw) in specs
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


@triton.autotune(configs=get_cuda_autotune_config(),
                 key=['M', 'N', 'K'],
                 prune_configs_by={"early_config_prune": _prune_configs},
                 post_hook=validate_results,
                 cache_results=True)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SWIZZLE_M: tl.constexpr):
    """
    C = A x B
    A: (M, K) float32
    B: (K, N) float32
    C: (M, N) float32
    """
    # Block index conversion
    pid_m, pid_n = pid_mn_from_pid(BLOCK_SIZE_M, BLOCK_SIZE_N, M, N, SWIZZLE_M)

    # Row indices in tile (pid_m, pid_n) in A and C
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    # Column indices in tile (pid_m, pid_n) in B and C
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    # Relative indices in K dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # BLOCK_SIZE_M × BLOCK_SIZE_K matrix with pointers to the A-tile elements
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    # BLOCK_SIZE_K × BLOCK_SIZE_N matrix with pointers to the B-tile elements
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # Initial BLOCK_SIZE_M × BLOCK_SIZE_N tile that this program instance will compute
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    mask_m = offs_am[:, None] < M
    mask_n = offs_bn[None, :] < N
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (k * BLOCK_SIZE_K + offs_k) < K
        # Load A and B tiles from global memory
        # Use masks to avoid out-of-bounds accesses
        a = tl.load(a_ptrs, mask=mask_m & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n, other=0.0)
        # accumulator += a @ b
        accumulator = tl.dot(a, b, accumulator)
        # Update pointers to the next A and B tiles
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    c = accumulator

    # Row and column indices of the C-tile
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # BLOCK_SIZE_M × BLOCK_SIZE_N matrix with pointers to the C-tile elements
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    # Mask to avoid out-of-bounds accesses
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
