import torch
import triton
import triton.language as tl
from triton.testing import do_bench

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
def mvm_kernel(a_ptr, b_ptr, c_ptr, M, N, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr):
    """
    C = A x B
    A: (M, K) float32
    B: (K, N) float32
    C: (M, N) float32
    """
    # pid: program ID
    # Corresponds to blockIdx.x in CUDA (axis=0)
    # Get the maximum number of program IDs: tl.num_programs(axis=0)
    #pid = tl.program_id(axis=0)

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
    '''
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
    '''

    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # modulo prevents overflow
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * N + offs_n[None, :])
    b_ptrs = b_ptr + (pid_n * BLOCK_SIZE_N + offs_n[:, None])
    
    accumulator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < N - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < N - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator

    offs_cn = pid * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptrs = c_ptr + (offs_cn[None, :])
    c_mask = (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def mvm(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, N = a.shape

    c = torch.empty((N), device=a.device, dtype=torch.float32)

    # Number of blocks = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
    # Number of threads per block: num_warps * 32
    # Each thread computes ≈ (BLOCK_SIZE_M * BLOCK_SIZE_N) / (num_warps * 32) elements of C
    # Define a 1-D grid of blocks
    grid = lambda META: ( \
        triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    time_ms = do_bench(lambda: mvm_kernel[grid](a, b, c, M, N))
    print(mvm_kernel.best_config)
    print("Runtime: " + str(time_ms))
    return c


# Test
torch.manual_seed(0)
#a = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
#b = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
a = (torch.rand((1024, 1024), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
b = (torch.rand((1024), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
triton_output = mvm(a, b)
torch_output = torch.mv(a, b)

# Compare outputs
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
    print("Test passed! ✅")
else:
    print("Test failed! ❌")
