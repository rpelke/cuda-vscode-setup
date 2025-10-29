import torch
import triton
import triton.language as tl


def check_and_launch_matmul(a, b, kernel, grid):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    kernel[grid](a, b, c, M, N, K)
    return c


# Helper: 1D -> 2D PIDs (using M-Swizzle)
@triton.jit
def pid_mn_from_pid(
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SWIZZLE_M: tl.constexpr,
):
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
    GROUP_SIZE = SWIZZLE_M * num_pid_n
    group_id = pid // GROUP_SIZE
    # Decrease swizzle in case M % (SWIZZLE_M * BLOCK_SIZE_M) != 0
    SWIZZLE_M_GRP = min(SWIZZLE_M, tl.cdiv((M - group_id * SWIZZLE_M * BLOCK_SIZE_M), BLOCK_SIZE_M))
    group_offs = SWIZZLE_M * group_id
    pid_m = ((pid % SWIZZLE_M_GRP) + group_offs)
    group_size_m = min(num_pid_m - group_offs, SWIZZLE_M)
    pid_n = (pid % GROUP_SIZE) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    return pid_m, pid_n
