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
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
        }, num_stages=5, num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 256,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 256,
            'BLOCK_SIZE_N': 128,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
        }, num_stages=4, num_warps=4),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def gemv_kernel(a_ptr, b_ptr, c_ptr, M, N, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr):
    """
    C = A x B
    A: (M, N) float32
    B: (N) float32
    C: (M) float32
    """

    pid_m = tl.program_id(axis=0)

    # determine the m-tile
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # modulo prevents overflow
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_am[:, None] * N + offs_n[None, :]    # 1 column with ptrs to A rows
    b_ptrs = b_ptr + offs_n

    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    # split summation per-element along N
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a = tl.load(a_ptrs, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)    # [None, :] broadcasts 1d array to 2d
        b = tl.load(b_ptrs, mask=offs_n < N - n * BLOCK_SIZE_N, other=0.0)
        accumulator += tl.sum(a * b[None, :], axis=1)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + (offs_cm)
    c_mask = (offs_cm < M)
    tl.store(c_ptrs, c, mask=c_mask)


def gemv(a, b):
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
    time_ms = do_bench(lambda: gemv_kernel[grid](a, b, c, M, N))
    print(gemv_kernel.best_config)
    print("Runtime: " + str(time_ms))
    return c


# Test
torch.manual_seed(0)
a = (torch.rand((1024, 1024), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
b = (torch.rand((1024), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
triton_output = gemv(a, b)
torch_output = torch.mv(a, b)

# Compare outputs
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
    print("Test passed! ✅")
else:
    print("Test failed! ❌")
