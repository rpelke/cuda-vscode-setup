import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 256,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'SWIZZLE_N': 8
        },
                      num_stages=4,
                      num_warps=4),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SWIZZLE_N: tl.constexpr):
    """
    C = A x B
    A: (M, K) float32
    B: (K, N) float32
    C: (M, N) float32
    """
    # pid: program ID
    # Corresponds to blockIdx.x in CUDA (axis=0)
    # Get the maximum number of program IDs: tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    # Maximum number of program IDs along M and N axes
    # tl.num_programs(axis=0) = num_pid_m * num_pid_n
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Translate 1D pid to 2D (pid_m, pid_n)
    # An example is shown below. For a more detailed explanation, see `triton.md`.
    #m\n |  0   1   2   3   4   5         m\n |  0    1    2    3    4    5
    #----+-----------------------         ----+-----------------------------
    # 0  |  0   2   4   6   8  10          0  | 0,0  0,1  0,2  0,3  0,4  0,5
    # 1  |  1   3   5   7   9  11    to    1  | 1,0  1,1  1,2  1,3  1,4  1,5
    # 2  | 12  14  16  18  20  22          2  | 2,0  2,1  2,2  2,3  2,4  2,5
    # 3  | 13  15  17  19  21  23          3  | 3,0  3,1  3,2  3,3  3,4  3,5
    GROUP_SIZE = SWIZZLE_N * num_pid_n
    group_id = pid // GROUP_SIZE
    group_offs = SWIZZLE_N * group_id
    pid_m = (pid % SWIZZLE_N) + group_offs
    group_size_m = min(num_pid_m - group_offs, SWIZZLE_N)
    pid_n = (pid % GROUP_SIZE) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Number of blocks = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
    # Number of threads per block: num_warps * 32
    # Each thread computes ≈ (BLOCK_SIZE_M * BLOCK_SIZE_N) / (num_warps * 32) elements of C
    # Define a 1-D grid of blocks
    grid = lambda META: ( \
        triton.cdiv(M, META['BLOCK_SIZE_M']) * \
        triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](a, b, c, M, N, K)
    return c


# Test
torch.manual_seed(0)
a = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
b = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)

# Compare outputs
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
    print("Test passed! ✅")
else:
    print("Test failed! ❌")
