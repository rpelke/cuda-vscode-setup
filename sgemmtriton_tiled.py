import torch
import triton
import triton.language as tl

@triton.jit
def sgemm_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    alpha, beta,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled k-loop
    for k0 in range(0, K, TILE_K):
        k = k0 + tl.arange(0, TILE_K)

        # Load tiles from A and B
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Outer product accumulation 
        acc += tl.dot(a, b)

    c = tl.load(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    c = alpha * acc + beta * c

    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def sgemm_tiled(A, B, C, alpha=1.0, beta=0.0):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"

    BLOCK_M, BLOCK_N = 64, 64   
    TILE_K = 32                

    grid = ((M + BLOCK_M - 1) // BLOCK_M,
            (N + BLOCK_N - 1) // BLOCK_N)

    sgemm_tiled_kernel[grid](
        A, B, C,
        M, N, K,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_K=TILE_K
    )
    return C

def ms_to_gflops(M: int, K: int, N: int, ms: float) -> float:
    """Convert runtime in milliseconds to GFLOPS for SGEMM (2*M*N*K operations)."""
    gflops = 2.0 * M * N * K / (ms * 1e6)
    return gflops

# Test
def main():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Matrix sizes
    M_0 = 1027
    N_0 = 1023
    K_0 = 1025
    alpha = 2.0
    beta = 3.0

    # Initialize matrices with random floats between 0 and 1
    A = torch.rand(M_0, K_0, device="cuda", dtype=torch.float32)
    B = torch.rand(K_0, N_0, device="cuda", dtype=torch.float32)
    C = torch.rand(M_0, N_0, device="cuda", dtype=torch.float32) 
    C_ref = C.clone()  

    # Run Triton SGEMM
    start.record()
    C_out = sgemm_tiled(A, B, C, alpha=alpha, beta=beta)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    print(f"SGEMM took {elapsed_ms:.3f} ms")
    
    gflops = ms_to_gflops(M_0, K_0, N_0, elapsed_ms)
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    # Compute expected result with PyTorch
    C_expected = alpha * (A @ B) + beta * C_ref
    print("Close to PyTorch:", torch.allclose(C_out, C_expected, atol=1e-8))
    
if __name__ == "__main__":
    main()
