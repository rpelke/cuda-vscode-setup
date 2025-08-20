# Branch Divergence and Predication

**Branch divergence** can have a big impact on GPU kernel performance.
On NVIDIA GPUs, threads run in warps of typically 32 threads that share a program counter (PC).
If all threads in a warp take the same branch (i.e., execute the same operations), execution is efficient.
But if threads **diverge**, e.g., in an if/else statement, the warp must execute each path serially.
Threads not on the current path are idle until they reach the **reconvergence point** again, reducing overall utilization.

A simple example of potential divergence is shown here:
```C++
__global__ void divergence_example() {
    if (cond) {
        f1(); // Execute f1
    } else {
        f2(); // Execute f2
    }
}
```

### Branch Efficiency
Branch efficiency measures the fraction of conditional branches where all threads in a warp took the same path:

$branch \ efficiency := 100 \cdot \frac{\\# branches - \\# divergent \ branches}{\\# branches}$

Let's measure this metric for our [warptiling](../src/06_sgemm_warptiling.cu) kernel.
It has many if/else statements to check whether the thread is out of bounds, e.g., for odd matrix dimensions such as:
```C++
int M_0 = 1027, N_0 = 1023, K_0 = 1025;
```

The [ncu metric](../readme.md#tracing) `sm__sass_average_branch_targets_threads_uniform`  SM.will give us the branch efficiency.
The results are:
| Metric | Value |
| -- | -- |
| sm__sass_average_branch_targets_threads_uniform.pct | $100\%$ |

Apperently, there is no "real" branch divergence in our kernel despite the `if/else` statements.
Why is this the case? This was a bit confusing at first because I was sure that there is at least one warp in my application that goes through both branches of an `if/else` statement.

The simple answer is that the compiler does optimizations that can replace branch instructions by **predicated instructions** for short, conditional code segments (see also [CUDA C Programming](https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf), p.86).


### Branch Predication and Masking
[Predictation](https://en.wikipedia.org/wiki/Predication_(computer_architecture)) is one way GPUs handle divergent control flow within a warp.
Instructions can be prefixed with a predicate, such as `@P0 INSTR`.
Here, `P0` is a predicate register.
Each thread has its own predicate bit, and a thread will only execute `INSTR` if its bit in `P0` is set to `1`.

With predication, there is no real branch divergence, since all instructions from all sides of the conditional are executed, but only the active threads update state.
The compiler applies predication only if the conditional body is short enough (below a certain instruction threshold).
For longer code paths, the compiler emits real branch instructions, which can lead to warp divergence if threads take different paths.

### Control Flow Efficiency
We have just seen that branch efficiency is $100\%$ despite predication.
However, there is a metric in which the effects of predication can be seen, called **control flow efficiency**.
Control flow efficiency measures the fraction of active threads for each executed instruction.

$control \ flow \ efficiency := \frac{\\# thread-instructions \ execute \ with \ predicate \ TRUE}{\\# thread-instructions \ executed}$

After profiling, we get the following results for the warptiling kernel:
| Metric | Value |
| -- | -- |
| sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime.pct | $98.94\%$ |
| sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime.ratio | $31.66$ (out of $32$) |

This means, on average $31.66$ out of $32$ threads (in a warp) actually executed the instruction because their predicate was true.


### Omitting Boundary Checks for Certain Kernels

For the following matrix dimensions, we are guaranteed not to run out of bounds, so boundary checks are unnecessary:

```C++
int M_0 = 1024, N_0 = 1024, K_0 = 1024;
```

I found it interesting to measure the run time cost of these `if/else` instructions, since in some cases they can be omitted, e.g., for the matrix dimension above (all threads always follow the same branch).

The measured run times are:
| Kernel | Run Time |
|--|--|
| Removed boundary checks | 0.801 ms |
| With boundary checks | 0.8205 ms |

For the tested warptiling kernel, the overhead is approximately $2.4\%$.
For performance critical applications, $2\%-3\%$ overhead might already be too much.
It can be optimized by using different kernel implementations for depending on the matrix dimension.


### Recognizing Predication in the SASS Code

Let's look at a simple code snippet.
The value 42 is only written to the positions of C whose index is divisible by 2.

```C++
float p = 42.0f;
if (threadIdx.x % 2 == 0) {
    C[threadIdx.x] = p;
}
```

The most important parts of the SASS code are as follows:
```x86asm <!-- No SASS highlighting. Use x86 assembly colors instead. -->
1   LOP3.LUT P6, RZ, R0, 0x1, RZ, 0xc0, !PT
2   @!P6 MOV R55, 0x42280000
3   @!P6 STG.E.SYS [R56], R55
```
In line 1, the active thread IDs written to the predicate `P6`.
Therefore, a LUT instruction is used.
`LUT3` has three inputs `A`, `B`, and `C` with
```
A = RZ (zero register)
B = RO (threadIdx.x)
C = 0x1
```
The executed LUT function is described by `0xc0`. When you look at the truth table, you can see that the LUT function corresponds to `B AND C`, hence `threadIdx.x AND 0x1`, which is `!(threadIdx.x % 2 == 0)`.
The resulting predicate bits are stored in `P6`.

| Index | A | B | C | LUT Function |
| ----- | - | - | - | ------- |
| 0     | 0 | 0 | 0 | 0       |
| 1     | 1 | 0 | 0 | 0       |
| 2     | 0 | 1 | 0 | 0       |
| 3     | 1 | 1 | 0 | 0       |
| 4     | 0 | 0 | 1 | 0       |
| 5     | 1 | 0 | 1 | 0       |
| 6     | 0 | 1 | 1 | 1       |
| 7     | 1 | 1 | 1 | 1       |

In line 2, the floating-point value `42.0f` (`0x42280000` in IEEE 754 standard) is moved into register `R55` but only if `!P6`, i.e., only for threads with an index divisible by two.
The last instruction in line 3, is a predicated store instruction to the global memory (`C[threadIdx.x]`).

-> The prefixed predicate conditions (here: `@!P6`) and the absense of branch instructions (e.g., `BRA`, `BSYNC`, `BSSY`) make it easy to recognize predication in SASS.


### Recognizing Branch Divergence in the SASS Code
To provoke branch divergence, we use an `if/else` example that is a little more complex.
We want to show that branches exist, so that the branch efficiency `sm__sass_average_branch_targets_threads_uniform.pct` drops below $100\%$.

**But be careful**: If the example is too simple, only predication is used. This means, no branch instructions are inserted. As a result, the branch efficiency is $0\%$ because there are no branches (see formula).

```C++
if (threadIdx.x % 2 == 0) {
    for(int i=0; i<100; i++) { C[threadIdx.x] += i; }
}
else {
    for(int i=0; i<100; i++) { C[threadIdx.x] -= i; }
}
```

The C++ code results in the following simplified SASS Code:

```x86asm <!-- No SASS highlighting. Use x86 assembly colors instead. -->
1     LDG.E.SYS R4, [R2]
2     LOP3.LUT R0, R0, 0x1, RZ, 0xc0, !PT
3     BMOV.32.CLEAR RZ, B0
4     BSSY B0, 0x719ccba55f30
5     ISETP.NE.U32.AND P0, PT, R0, 0x1, PT
6     @P0  BRA 0x719ccba558e0 (here: 107)
7     FADD R4, R4, -1
      ...
105   FADD R5, R4, -99
106   BRA 0x719ccba55f20 (here: 207)
107   FADD R4, RZ, R4
108   FADD R4, R4, 1
      ...
206   FADD R5, R4, 99
207   BSYNC B0
208   STG.E.SYS [R2], R5
```

In line `1`, the value of `C[threadIdx.x] ` is loaded into `R4`.
Line `2` and line `5` program the predicate register `P0` according to the condition `threadIdx.x % 2 == 0`.
Line `3` clears the reconvergence stack. `B0` (... `B7`) are reconvergence registers.
They hold the address of the reconvergence point.
Line `4` sets a new reconvergence point to address `207`.
All threads are reunited at this address.

Line `6` lets all threads that fulfill `threadIdx.x % 2 == 0` jump to `107`.
These threads execute the instructions inside the `if` body.
If the condition is not fulfilled, the odd threads execute the instructions inside the `else` body starting from line `7`. In line `106` they jump to line `207`.
Finally, the value is written back to `C[threadIdx.x]` in line `208`.
Profiling shows that the number of branch instructions ` sm__sass_branch_targets` is $>0$ and the branch efficiency ` sm__sass_average_branch_targets_threads_uniform` is $<100\%$.

-> Branch divergence can easiliy be recognized by the typical branch instructions `BRA`, `BSYNC`, and `BSSY`.
