import torch
import triton
import triton.language as tl
from triton.testing import do_bench

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_M': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 64,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 128,
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 256,
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 32,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 32,
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 32,
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 16,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 16,
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 16,
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 8,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 8,
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 8,
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            'BLOCK_SIZE_M': 4,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 4,
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 4,
        },
                      num_stages=5,
                      num_warps=2),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)
    for row_idx in tl.range(row_start, n_rows, row_step): #num_stages controls unrolling of this loop
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 # TODO: remove, redundant

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Number of blocks = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
    # Number of threads per block: num_warps * 32
    # Each thread computes ≈ (BLOCK_SIZE_M * BLOCK_SIZE_N) / (num_warps * 32) elements of C
    # Define a 1-D grid of blocks
    #grid = lambda META: ( \
    #    triton.cdiv(M, META['BLOCK_SIZE_M']) * \
    #    triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    #time_ms = do_bench(lambda: softmax_kernel[(16, 1, 1)](c, a, a.stride(0), c.stride(0), a.shape[0], a.shape[1], a.shape[1], 2))
    grid = lambda META: ( \
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        1 )
    
    time_ms = do_bench(lambda: softmax_kernel[grid](c, a, a.stride(0), c.stride(0), a.shape[0], a.shape[1], a.shape[1]))
    #time_ms = do_bench(lambda: softmax_kernel[grid](c, a, a.stride(0), c.stride(0), a.shape[0], a.shape[1], a.shape[1]))
    #ptx = triton.compile(softmax_kernel.asm)
    #print(ptx)
    print(softmax_kernel.best_config)
    print("Runtime: " + str(time_ms))
    return c


# Test
torch.manual_seed(0)
#a = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
#b = (torch.rand((512, 512), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
a = (torch.rand((1024, 1024), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
b = (torch.rand((1024, 1024), device=DEVICE, dtype=torch.float32) - 0.5).contiguous()
triton_output = matmul(a, b)
torch_output = torch.softmax(a, 0)

# Compare outputs
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
    print("Test passed! ✅")
else:
    print("Test failed! ❌")

print(triton_output)
print(torch_output)