# Register Blocking and Instruction Level Parallelism (ILP)
In this experiment, I want to take a closer look at register blocking (also called register tiling) and instruction-level parallelism (ILP).
Both optimization techniques are closely related because they operate within a single thread.
Please read the basics of [ILP](00_instruction_level_parallelism.md) first.

**Register blocking**: Registers are the fastest available memory on the GPU. Values should be loaded from slower memory (like global or shared memory) into registers and reused as much as possible to avoid redundant memory accesses and to maximize the ratio of arithmetic operations per memory access (**arithmetic intensity**).


## Implementation of a Mini-GEMM inside a Single Thread
Let's look at the mini-GEMM operation executed by a single thread within the [warptiling](../src/06_sgemm_warptiling.cu) kernel. The computation produces a result matrix `tmp` of shape $TM \times TN$, computed from two input matrices `As` ($TM \times BK$) and `Bs` ($BK \times TN$), both located in shared memory and stored in row-major format.
While iterating over the `BK` axis:

- Values in `As` are accessed contiguously (good locality).
- Values in `Bs` are accessed non-contiguously.

I call this operation 'mini'-GEMM because the idea is that `TM`, `TN`, and `BK` are so small that we can keep the matrix values in registers and reuse them.


### Implementation 1: Nested Loops
The first implementation uses a naive 3-level nested loop. For each triplet `(tm, tn, bk)`, one value is read from As and one from `Bs`, and immediately used in the accumulation. There is no reuse of values across iterations.

-> Simple & easy to read but: No value reuse; memory pipline heavily used.

```C++
float tmp[TM_06][TN_06] = {0.0f};
for (int tm = 0; tm < TM_06; ++tm) {
    for (int tn = 0; tn < TN_06; ++tn) {
        for (int bk = 0; bk < BK_06; ++bk) {
            tmp[tm][tn] += As[BK_06 * (WM_06 * wy + TM_06 * ty + tm) + bk] *
                            Bs[BN_06 * bk + WN_06 * wx + TN_06 * tx + tn];
        }
    }
}
```

### Implementation 2: Load Shared Memory Values from As and Bs only once
The second implementation improves memory reuse by first loading slices of `As` and `Bs` into registers before performing any computation. For each `bk`:

- All required TM values from `As` are loaded into `reg_A[]`.
- All required TN values from `Bs` are loaded into `reg_B[]`.
- The outer product is computed using these cached values.

-> Reuses shared memory values, increases register usage, potential for instruction-level parallelism (ILP).

```C++
float tmp[TM_06][TN_06] = {0.0f};
for (int bk = 0; bk < BK_06; ++bk) {
    float reg_A[TM_06];
    #pragma unroll
    for (int tm = 0; tm < TM_06; ++tm) {
        reg_A[tm] = As[BK_06 * (WM_06 * wy + TM_06 * ty + tm) + bk];
    }
    float reg_B[TN_06];
    #pragma unroll
    for (int tn = 0; tn < TN_06; ++tn) {
        reg_B[tn] = Bs[BN_06 * bk + WN_06 * wx + TN_06 * tx + tn];
    }
    #pragma unroll
    for (int tm = 0; tm < TM_06; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TN_06; ++tn) {
            tmp[tm][tn] += reg_A[tm] * reg_B[tn];
        }
    }
}
```

## Implementation 1 vs. Implementation 2 with -O0
For the first comparison, we will turn off all compiler optimizations, so that we can better evaluate the influence of our otimizations (spoiler: the compiler aggressive optimizations (O3) are better than my own optimizations).
CUDA code goes through two compilation stages, so you need to disable optimizations at both:

```
CUDA source code (.cu)
        ▼
      nvcc
(compiler driver)
        ▼
       PTX
(intermediate assembly)
        ▼
      ptxas
(PTX assembler)
        ▼
GPU machine code (SASS)
```

**nvcc** handles host code (CPU) and orchestrates the compilation of device code (GPU) into PTX.
**ptxas** converts PTX (Parallel Thread eXecution) intermediate code into GPU machine code (SASS).

To switch optimizations off in the [CMakeLists.txt](../CMakeLists.txt), use:
```bash
# nvcc
set(CMAKE_CUDA_FLAGS -O0)
# ptxas
target_compile_options(sgemm PRIVATE
    ...
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O0>
)
```

Make sure that tracing is enabled in the [settings.json](../.vscode/settings.json):
```json
"CMAKE_ENABLE_TRACING": "ON"
```

For `BM_06=64`, `BN_06=32`, `TM_06=4`, `TN_06=2`, `BK_06=WN_06=WM_06=16`, and `M_1=N_1=K_1=1024`, I get the following results:

| Implementation | Optimization Level | Run time |
|--|--|--|
| 1 | O0 | 2.48 ms|
| 2 | O0 | 1.74 ms|

As expected, the second implementation is faster because it better reuses the values from shared memory.
To get more insights, `ncu-ui` can be used for tracing.
The [readme.md](../readme.md#tracing) explains how it works.

When we look at an extract of sass instructions inside the `bk` loop, we can see a bunch of `LDS` (load), `IADD3` (add immediate), and `MOV` (move) instruction followed by one or two `FFMA` (floating-point fused multiply accumulate) instructions.
We can see 8 `FFMA` instructions, one for every element of the `TM_06 x TN_06` matrix.
While the values of the `reg_B` array are held in `R41` and `R43`, only one register is used for `reg_A` (`R42`).

```x86asm <!-- No SASS highlighting. Use x86 assembly colors instead. -->
IADD3 R41, R35, -0x100, RZ
MOV R41, R41
LDS.U R41, [R41]
IADD3 R42, R2, -0x80, RZ
MOV R42, R42
LDS.U R42, [R42]
FFMA R40, R42, R41, R40
IADD3 R43, R35, -0xfc, RZ
MOV R43, R43
LDS.U R43, [R43]
FFMA R39, R42, R43, R39
IADD3 R42, R2, -0x40, RZ
MOV R42, R42
LDS.U R42, [R42]
FFMA R38, R42, R41, R38
FFMA R37, R42, R43, R37
MOV R2, R2
LDS.U R42, [R2]
FFMA R36, R42, R41, R36
FFMA R34, R42, R43, R34
IADD3 R42, R2, 0x40, RZ
MOV R42, R42
LDS.U R42, [R42]
FFMA R33, R42, R41, R33
FFMA R32, R42, R43, R32
```

## Implementation 1 vs. Implementation 2 with Optimizations
Now, lets compare the run times which optimizations switched on.
With `O3`, nested loops get aggressively unrolled, reordered, and optimized.
Instead of the previous load-use-load-use pattern, we can see a batch of load operations first, follwed by a batch of FFMAs. This is done to hide the load latency.
Moreover, we can see vectorized load operations `LDS.U.64` and `LDS.U.128`.
In the snipped below, the 64-bit operations belong to loads from `Bs` (TN=2) and 128-bit operations belong to loads from `As` (TM=4).
For example, the load operaion `LDS.U.128 R20, [R4]` loads to registers R20-R23.

```x86asm
LDS.U.64 R26, [R10.X4+0x1000]
LDS.U.128 R20, [R4]
LDS.U.64 R24, [R10.X4+0x1080]
LDS.U.64 R40, [R10.X4+0x1100]
LDS.U.128 R12, [R4+0x40]
LDS.U.128 R36, [R4+0x80]
LDS.U.128 R16, [R4+0xc0]
LDS.U.64 R42, [R10.X4+0x1180]
FFMA R28, R26, R20, RZ
FFMA R20, R20, R27, RZ
LDS.U.128 R32, [R4+0x50]
FFMA R29, R24, R21, R28
FFMA R21, R21, R25, R20
FFMA R20, R40, R22, R29
FFMA R22, R22, R41, R21
FFMA R21, R26, R12, RZ
FFMA R12, R12, R27, RZ
FFMA R21, R24, R13, R21
FFMA R28, R26, R36, RZ
FFMA R44, R36, R27, RZ
FFMA R12, R25, R13, R12
FFMA R45, R26, R16, RZ
FFMA R16, R16, R27, RZ
FFMA R13, R24, R37, R28
FFMA R45, R24, R17, R45
```

When comparing the run times of implementation 1 and 2, we don't see a difference anymore.
This means, especially when the loop boundaries are know at compile time, the compiler will likely find better optimizations than we will do manually.

| Implementation | Optimization Level | Run time |
|--|--|--|
| 1 | O0 | 0.8 ms|
| 2 | O0 | 0.8 ms|

Moreover, the auto-vectorization of the load accesses also explains, why our manually vectorized kernel [05_sgemm_tiled_2d_vectorized_2.cu](../src/05_sgemm_tiled_2d_vectorized_2.cu) is not faster than [03_sgemm_tiled_2d.cu](../src/03_sgemm_tiled_2d.cu).
A look at the sass assembly of the latter confirms this hypothesis: the load accesses are already vectorized.
