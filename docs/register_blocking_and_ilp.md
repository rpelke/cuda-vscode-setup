# Register Blocking and Instruction Level Parallelism (ILP)
In this experiment, I want to take a closer look at register blocking (also called register tiling) and instruction-level parallelism (ILP).
Both optimization techniques are closely related because they operate within a single thread.

- **Register blocking**: Registers are the fastest available memory on the GPU. Values should be loaded from slower memory (like global or shared memory) into registers and reused as much as possible to avoid redundant memory accesses.
- **ILP**: Independent consecutive instructions can be executed in parallel.

To be honest, the concept of ILP confused me at first because I initially knew it only in the context of [superscalar processors](https://en.wikipedia.org/wiki/Superscalar_processor).
In the context of CPUs, ILP usually means that two or more instructions can be executed in parallel on different duplicate ALUs but only if there are no data dependencies between them.
This idea also applies to GPUs. Most modern GPUs have multiple functional units, such as for integer and floating-point arithmetic. These can be used in parallel by different instructions if the hardware and scheduler allow it.

However, there's another closely related concept: achiving a high **issue rate**.
This is analogous to reducing pipeline stalls on CPUs. A CPU pipeline typically consists of several stages (in a pipelined processor). A classic 5-stage pipeline includes:
Fetch, Decode, Execute, Memory Access, and Write Back.
This means that, ideally, 5 different instructions are being processed simultaneously, but each in a different stage.

**Stalling** happens when some instructions depend on the results of earlier ones. In that case, the CPU must pause parts of the pipeline to wait for data, which reduces instruction throughput and increases latency.
A common type of dependency is a read-after-write (RAW) hazard, where instruction 2 needs a value computed by instruction 1. To maintain correctness, the processor must ensure that instruction 2 doesn't read the wrong (old) value, potentially requiring a pipeline stall.
So, both hardware and software aim to avoid such stalls to keep the pipeline running smootly.
On the software side, compilers can perform optimizations such as **loop unrolling** and **instruction reordering** to increase instruction-level parallelism and reduce dependencies.
On the hardware side, techniques like **bypassing** (also called **forwarding**) are used. These allow intermediate results to be forwarded directly from a functional unit's output back to its input without taking a detour through a register file.

The same ideas apply to GPUs. Let's look more closely at what happens inside a Streaming Multiprocessor (SM).
Therefore, we'll take a closer look at the [Turing Architecture](https://en.wikipedia.org/wiki/Turing_(microarchitecture)), since I'll later benchmark on an NVIDIA GeForce RTX 2080 Ti.

According to the offical Turing [documentation](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf):

> "The Turing SM supports concurrent execution of FP32 and INT32 operations. [...] The Turing SM is partitioned into four processing blocks, each with 16 FP32 Cores, 16 INT32 Cores, two Tensor Cores, one warp scheduler, and one dispatch unit."

Each warp scheduler is responsible for choosing which warp to execute next based on readiness (e.g., no pending dependencies), data availability, and whether the required functional units are free.
The dispatch unit sends the selected instruction to a specific execution path, commonly called a 'pipeline' or 'pipe', that is optimized for a certain instruction type, such as FP32, INT32, or load/store operations.

Moreover, the [tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html?utm_source=chatgpt.com) provides several relevant insights:

- Instructions are performed over **two** cycles.
- The schedulers can issue independent instructions **every** cycle.
- Dependent instruction issue latency for core FMA math operations **four** clock cycles. 

And then it concludes: Execution latencies of math operations can be hidden
1. "by as few as 4 warps per SM, assuming 4-way instruction-level parallelism ILP per warp."
1. "by 16 warps per SM without any instuction-level parallelism."

At first, this seemed contradictory to me.
On one hand, dependent instructions have an issue latency of 4 cycles, and the scheduler can issue one instruction per cycle.
On the other hand, each instruction requires two cycles, since only 16 cores (FP32 or INT32) are available per warp scheduler, but a warp has 32 threads.

How can a single warp issue 4 instructions in 4 cycles if each instruction takes 2 cycles?
Here's how I interpret this information (correct me if I'm wrong):

1. Each warp scheduler can issue one instruction per cycle, as long as the instructions are independent and target different functional units.
For example: `INT32 -> FP32 -> INT32 -> FP32` works.
But `4x INT32` in a row does not work since one instruction requires two cycles.
1. For the same math pipe, e.g., INT32, the schedular can issue a new instrcution every 2 cycles.
1. An SM has 64 FP32 and 64 INT32 cores in total. It can execute, e.g., four INT32 operations in parallel, but only if these operations belong to four different warps, each scheduled by a different warp scheduler.
