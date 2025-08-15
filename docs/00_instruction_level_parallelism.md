# Instruction Level Parallelism (ILP)
Instruction-Level Parallelism (ILP) is a fundamental concept in GPU programming.
The core idea is that multiple instructions from a warp can be executed simultaneously.


### ILP on GPUs vs. Superscalar CPUs
The concept of ILP is not new.
You might have heard about it in the context of [superscalar processors](https://en.wikipedia.org/wiki/Superscalar_processor) before.
There, it usually means that two or more instructions can be executed in parallel on different "duplicated" ALUs but only if there is no data dependency between them.
This idea also applies to GPUs. Most GPUs have multiple functional units, such as for integer and floating-point arithmetic. These can be used in parallel by different instructions.


### Instruction Issue Rate in GPUs and CPUs
Closely related to ILP is the goal of achieving a high **instruction issue rate**:
the rate at which new instructions are sent into execution.

In CPUs, this usually means keeping the CPU pipeline fed with new instructions every cycle to avoid stalls caused by dependencies or resource conflicts.
A CPU pipeline typically consists of several stages (in a pipelined processor). A classic 5-stage pipeline includes:
Fetch, Decode, Execute, Memory Access, and Write Back.
This means that, ideally, 5 different instructions are being processed simultaneously, but each in a different stage.

**Stalling** happens when some instructions depend on the results of earlier ones. In that case, the CPU must pause parts of the pipeline to wait for data, which reduces instruction throughput and increases latency.
A common type of dependency is a read-after-write (RAW) hazard, where instruction 2 needs a value computed by instruction 1. To maintain correctness, the processor must ensure that instruction 2 doesn't read the wrong (old) value.
So, both hardware and software aim to avoid such stalls to keep the pipeline running smootly.
On the software side, compilers can perform optimizations such as **loop unrolling** and **instruction reordering** to increase instruction-level parallelism and reduce dependencies.
On the hardware side, techniques like **bypassing** (also called **forwarding**) are used. These allow intermediate results to be forwarded directly from a functional unit's output back to its input without taking a detour through a register file.

GPUs work on the same basic idea:
once an instruction is issued, the scheduler can either send another **independent instruction from the same warp** (ILP) or **switch to a different warp** that has no pending dependencies (Thread-Level Parallelism (TLP)).


### Instruction Issuing, Dispatching, and Instruction Latency in a Streaming Multiprocessor (SM)
Let's look more closely at what happens inside a Streaming Multiprocessor (SM).
Therefore, we'll take a closer look at the [Turing Architecture](https://en.wikipedia.org/wiki/Turing_(microarchitecture)), since I'll later benchmark on an NVIDIA GeForce RTX 2080 Ti.

According to the offical Turing [documentation](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf):

> "The Turing SM supports concurrent execution of FP32 and INT32 operations. [...] The Turing SM is partitioned into four processing blocks, each with 16 FP32 Cores, 16 INT32 Cores, two Tensor Cores, one warp scheduler, and one dispatch unit."

Each warp scheduler is responsible for choosing which warp to execute next based on readiness (e.g., no pending dependencies), data availability, and whether the required functional units are free.
That's called **issuing** instruction(s).
The **dispatch** unit sends the selected instruction to a specific execution path, commonly called a 'pipeline' or 'pipe', that is optimized for a certain instruction type, such as FP32, INT32, or load/store operations.
Some GPU architectures, like the Turing architectures, are able to dispatch 2 instructions per cycle.
That's called **dual issue**.
However, these two instructions are not arbitrary. Only certain combinations such as INT32 and FP32 are possible.

Furthermore, it is important to understand that instruction issues always occur in a cycle but instruction dispatch can take several cycles.
The number of cycles per dispatch is often determined by the number of functional units available to the scheduler and the warp size. In our case, e.g., for INT32 instructions: $32\frac{threads}{warp} / 16~\mathrm{INT32~cores}$.
This leads to 2 dispatch cycles per INT32 instruction.

However, this does not mean that the **instruction latency** is 2 cycles.
The latency of an instruction is the cycle time from dispatch until the result is ready,
i.e., the result is stored. in the register file or memory.
Latency can be quite high, especially for load and store operations, in particular if you have to access DRAM instead of caches. But if there are no dependencies, the scheduler does not have to wait for the instruction to finish unil issuing the next instruction.


### Some Examples
To understand this, I found the examples from [this lecture](https://www.ece.lsu.edu/gp/notes/set-nv-org.pdf) quite helpful.
One example for a Pascal architecture (6.1) from the lecture is shown in the following.
The architecture is single issue and has 16 LS (Load/Store) and 32 FP32 and INT32 units per scheduler.
The load latency is approximately 400 cycles and other latencies are 6 cycles.

| Nr.   | Instruction                      | $t_{is}$ | Dependency | $t_{re}$ |
|-------|----------------------------------|----------|------------|----------|
| .L_2  |                                  |          |            |          |
| I00:  | MOV R2, R6                       | 0        |            |  6       |
| I01:  | MOV R4, R8                       | 1        |            |  7       |
| I02:  | LD.E R2, [R2]                    | 6        | R2         |  406     |
| I03:  | IADD32I R8.CC, R8, 0x4           | 8        | R8         |  14      |
| I04:  | MOV R5, R9                       | 9        |            |  15      |
| I05:  | IADD32I R0, R0, 0x1              | 10       |            |  16      |
| I06:  | ISETP.GE.AND P0, PT, R0, R11, PT | 16       | R0         |  22      |
| I07:  | IADD.X R9, RZ, R9                | 17       |            |  23      |
| I08:  | IADD32I R6.CC, R6, 0x4           | 18       |            |  24      |
| I09:  | IADD.X R3, RZ, R3                | 24       | CC         |  30      |
| I10:  | FADD R7, R2, 1                   | 406      | R2         |  412     |
| I11:  | ST.E [R4], R7                    | 412      | R7         |          |
| I12:  | @!P0 BRA (.L_2)                  | 414      |            |          |

The issue time is written as $t_{is}$ and the ready time is written as $t_{re}$.
The issue/dispatch diagram for a single warp $wp0$ looks like:
| #Cycle |0|1|2|...|5|6|7|8|9|10|...|16|17|18|...|24|...|406|...|412|413|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| wp0 |I0|I1||||[I2|]|I3|I4|I5||I6|I7|I8||I9||I10||[I11|]|

If the dispatch takes more than a single cycle, brackets ([ ]) where used.

Now, lets consider a dual issue SM with two warps $wp0$ and $wp1$ for a single scheduler.
The scheduler has 16 FP32 units, 16 INT units, and 32 LS units.
The instructions for each thread are: `INT0 -> FP0 -> INT1 -> FP1 -> LS0 -> LS1`.
All instructions are independent.
The resulting issue/dispatch graph looks like:

| #Cycle |0|1|2|3|4|5|6|7|8|9|10|11|
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| wp0 | [INT0 | ] |       |   | [INT1 | ] |       |   | LS0 |     | LS1 |     |
|     | [FP0  | ] |       |   | [FP1  | ] |       |   |     |     |     |     |
| wp1 |       |   | [INT0 | ] |       |   | [INT1 | ] |     | LS0 |     | LS1 |
|     |       |   | [FP0  | ] |       |   | [FP1  | ] |     |     |     |     |


## Summary: Issuing/Dispatching
1. The scheduler choose a warp that is
    - **Active**: it is loaded to the SM and blocks resources.
    - **Ready**: not waiting for memory or register operands.
1. One or two instructions from the warp (depending on the architecture) are issued to the dispatcher.
1. Instructions are fetched and decoded. Depending on the number of functional units $f$, multiple cycles may be needed to dispatch a single instructions.
1. At each cycle, $f$ threads are dispatched to functional units.


### NVIDIAs Tuning Guide
Coming back to our Turing architecture, the [tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/#instruction-scheduling) provides several relevant insights:

- **Dispatch**: Instructions are dispatched over **two** cycles to a given functional unit (e.g., FP32), because each warp scheduler has only 16 cores of that type but a warp has 32 threads.
- **Issue**: The schedulers can issue independent instructions **every** cycle.
- **Latency**: Dependent instruction issue latency for core FMA math operations is **four** clock cycles.

And then it concludes:
> "Execution latencies of core math operations can be hidden by as few as **4 warps per SM, assuming 4-way ILP per warp**, or by **16 warps per SM without any ILP**."

At first, this conclusion seemed confusing to me.
On one hand, dependent instructions have an issue latency of 4 cycles, and the scheduler can issue one instruction per cycle. On the other hand, each instruction requires two cycles to dispatch to the same pipeline.
Why are two independent FMA instructions not sufficient to hide the latency of the first one?

I assume that in this context, NVIDIA defines latency hiding as "the scheduler can on average issue one instruction per cycle", regardless of which functional unit it goes to, and regardless of dual issue capabilities.
This means, the scheduler can on average issue one instruction per cycle without having to stall because the FMA result is not ready yet.

For example, the latency of `FFMA0` can be hidden if we execute four independent instructions (two FP32 FMAs and two INT32 ops) in the following order. This keeps the scheduler busy every cycle:

| #Cycle | 0      | 1      | 2      | 3      |
|--------|--------|--------|--------|--------|
| wp0    | [FFMA0 |    ]   | [FFMA1 |     ]  |
|        | [INT0  |    ]   | [INT1  |     ]  |

Another possibility is having four warps (each mapped to its own FP32 unit), so that the scheduler also issues one instruction per cycle:

| #Cycle | 0      | 1      | 2      | 3      | 4      |
|--------|--------|--------|--------|--------|--------|
| wp0    | [FFMA0 |      ] |        |        |        |
| wp1    |        | [FFMA0 |      ] |        |        |
| wp2    |        |        | [FFMA0 |      ] |        |
| wp3    |        |        |        | [FFMA0 |      ] |
