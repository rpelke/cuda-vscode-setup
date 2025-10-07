# How Triton Works (Compare to CUDA C++) 
Triton is an open-source language and compiler for writing custom GPU kernels directly in Python. You write a ''single-program'' kernel with `triton.jit`, launch it over a grid of program instances (similar to CUDA thread blocks), and Triton compiles it Just-In-Time to highly optimized GPU code.


### Organization (and Distribution) of the Workload in Triton
Let's look at the CUDA C++ [warptiling](../../src/06_sgemm_warptiling.cu) implementation again.
Like CUDA C++, you must decide how to tile the problem and how many blocks to launch,
but Triton operates at a higher abstraction level.
For a GEMM kernel in Triton, you typically choose:

* Tile sizes: `BLOCK_SIZE_M`, `BLOCKSIZE_N`, and `BLOCKSIZE_K`
* Grid size and dimension, e.g.:
`grid = (ceil(M/BLOCK_SIZE_M) * ceil(N/BLOCK_SIZE_N),)`
for a 1D (linearized 2D) grid or
`grid = (ceil(M / BLOCK_SIZE_M), ceil(N / BLOCK_SIZE_N))`
for a 2D grid.
* Number of warps per block: `num_warps`
* The resulting number of thread per block is: `num_warps * 32`
* The depths of the software pipeline over the K-tiles: `num_stages`. This is explained [here](#understanding-num_stages).

Unlike CUDA C++, Triton handles the following tasks for you:

* The distribution of work within a block among the threads/warps 
(which output elements each thread computes), i.e.,
what you manually encoded with warp tiles (`WM`,`WN`) and register tiles (`TM`/`TN`) in the
[warptiling](../../src/06_sgemm_warptiling.cu) kernel.
* Low-level details inside a thread: loop unrolling, FMA loops, tensor-core MMA instructions when applicable, etc.


### Understanding `num_stages`
Consider a block computing one `BM`x`BN` tile of `C` (see [warptiling figure](../../docs/figures/06_warptiling_sgemm.png)).
In a classic CUDA warptiling kernel, each K-slice of A/B (size `BM`x`BK` and `BK`x`BN`) is loaded into shared memory, then used to update the accumulators.

Pseudo code for single-stage execution:

```C++
// Shared memory for the two tiles from A and B
__shared__ float As[BM * BK], Bs[BK * BN];

for (int k=0; k < K; k += BK) {
    // Global memory -> shared memory
    load_tile(As, Bs, k);
    __syncthreads();

    // Compute matmul operation: each thread stores results into tmp registers
    compute_tile(tmp, As, Bs);
    // Update result matrix C: C += tmp
    update_C(tmp);
    __syncthreads();
    ...
}
```
This is effectively `num_stages = 1` (load then compute).
`L` stands for `Load` and `C` stands for `Compute`.

| #Stages | k=0     | k=1     | k=2     |
|---------|---------|---------|---------|
| 1       | L0 + C0 | L1 + C1 | L2 + C2 |

A 2-stage pipeline (double-buffering) prefetches the next K-tile while computing on the current one:

```C++
__shared__ float As0[BM * BK], Bs0[BK * BN];
__shared__ float As1[BM * BK], Bs1[BK * BN];

int k = 0;
// Pre-loop: load tiles for k=0 into shared memory
load_tile(As0, Bs0, k);
__syncthreads();

for (; k < K; k += BK) {
    // Prefetch the next tile (k+BK). On Ampere+: using cp.async
    prefetch_tile(As1, Bs1, k + BK);

    // Compute matmul for the current tile (k)
    compute_tile(tmp, As0, Bs0);
    // Update result matrix C: C += tmp
    update_C(tmp);

    // Swap buffers
    swap(As0, As1); swap(Bs0, Bs1);
    __syncthreads();
    ...
}
```

On newer GPUs (Ampere+), the prefetch can be asynchronous (`cp.async`), also called non-blocking, so loading for `k+BK` overlaps with compute for `k`.
Setting a higher `num_stages` tells Triton's compiler to pipeline the K-loop.
This issues load operations for future tiles early and overlapps them with compute.

| #Stages | Pre-loop | k=0     | k=1     | k=2     |
|---------|----------|---------|---------|---------|
| 1       |          | L0 + C0 | L1 + C1 | L2 + C2 |
| 2       | L0       | L1 + C0 | L2 + C1 | L3 + C2 |
| 3       | L0 + L1  | L2 + C0 | L3 + C1 | L4 + C2 |

More stages lead to a better overlap.
However, this comes at the cost of a higher requirement for registers per thread (for A/B fragment base address, masks, addresses, loop variables, etc.) and a higher requirement for shared memory.
Therefore, too many stages can lead to register spills and reduced occupancy.


### Occupancy
A warp is considered active from the time its threads begin executing to the time when all threads in the warp have exited from the kernel. There is a maximum number of warps which can be concurrently active on a Streaming Multiprocessor (SM). Occupancy is defined as the ratio of active warps on an SM to the maximum number of active warps supported by the SM. Occupancy varies over time as warps begin and end, and can be different for each SM
[[Source]](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm).

The number of warps per SM cannot be chosen directly by the programmer but indirectly by the choice of the workload distribution, register requirements per thread (register file is shared among all active threads on an SM), shared memory per block etc.
For memory-bound applications, it is nice to have a high occupancy to hide the load/store latency effectively.
For compute-bound applications, a high occupancy is not so important (sometimes even bad [[Source]](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf)) because a smaller number of warps per SM can already utilize the ALU pipelines.
