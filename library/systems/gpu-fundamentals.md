# GPU Fundamentals

*Why GPUs are fast, why memory is the bottleneck, and how to reason about performance.*

---

## Why GPUs for deep learning

A CPU has tens of cores tuned for complex, sequential, branchy work. A GPU has thousands of simpler cores tuned to run the **same operation across lots of data at once** — exactly the shape of dense linear algebra.

**The dominant operation is matrix multiply (GEMM).** Attention is GEMMs; the MLP block is two GEMMs. Nearly all the FLOPs in a transformer are GEMMs, which is why hardware and kernels are built around them.

---

## The memory hierarchy

From fastest/smallest to slowest/largest (order-of-magnitude figures, Hopper-class):

| Level | Size | Speed |
|---|---|---|
| Registers (per thread) | < 1 KB | fastest (~1 cycle) |
| Shared memory / L1 (per SM) | ~228 KB | ~30 cycles |
| L2 cache | ~50 MB | slower |
| HBM (GPU RAM) | 80 GB (H100) | ~3.35 TB/s |
| CPU RAM | 100s of GB | much slower (over PCIe / NVLink-C2C) |
| NVMe / disk | TBs | slowest |

**The core tension:** an H100 SXM has ~990 TFLOPS of FP16 *Tensor Core* compute but ~3.35 TB/s of HBM bandwidth. Compute grows far faster than bandwidth across generations, so feeding the cores is the recurring problem.

For an N×N matrix multiply:
- **Compute:** O(N³) operations
- **Memory:** O(N²) elements read

For large N, compute dominates reads → **compute-bound**.

For element-wise ops (activation, normalization):
- **Compute:** O(N) operations
- **Memory:** O(N) elements

→ **memory-bound** — limited by read/write bandwidth, not math.

---

## The roofline model

**Arithmetic intensity** (FLOPs per byte moved) tells you which regime an op lives in:

```
Performance = min(Peak FLOPS, Bandwidth × Arithmetic Intensity)
```

- **High intensity** (large GEMM) → compute-bound → you can saturate the FLOPS.
- **Low intensity** (element-wise ops) → memory-bound → you can only saturate bandwidth.

Plot performance vs. arithmetic intensity: ops under the sloped bandwidth line are memory-bound; ops under the flat peak-FLOPS line are compute-bound. The "ridge point" where the two meet is the intensity you must exceed to be compute-bound.

---

## CUDA programming model (basics)

A kernel launches a **grid** of **blocks** of **threads**:

- **Thread** — smallest unit; runs one instance of the kernel.
- **Block** — group of threads that share **shared memory** and can synchronize.
- **Grid** — all blocks in the launch.

Threads run in **warps** (32 threads) in lockstep — **SIMT** (Single Instruction, Multiple Thread). If threads in a warp take different branches, the GPU runs both paths serially with inactive lanes masked off → **warp divergence**, a common performance killer.

**Shared memory** is a programmer-managed scratchpad (same SRAM as L1). The winning pattern: load a tile from HBM into shared memory once, do all the work there, write the result back once. This is the mechanism behind FlashAttention.

---

## Why FlashAttention matters

Standard attention materializes the full N×N score matrix in HBM:

1. `S = QKᵀ` — write S to HBM (N² elements)
2. `P = softmax(S)` — read S, write P (2×N² traffic)
3. `O = PV` — read P (N² reads)

That is O(N²) HBM memory and O(N²) traffic. At long context (e.g. 128K tokens) it is both slow and memory-hungry.

**FlashAttention** fuses all three steps into one kernel via **tiling**: keep the intermediate score/probability tiles in on-chip SRAM and never write the full matrix to HBM (online softmax keeps the running normalizer). This is **IO-aware** computation.

Result: identical output, O(N) HBM memory, and a large wall-clock speedup. The math is unchanged — only the memory access pattern is. FlashAttention-2/3 push this further with better work partitioning and, on Hopper, FP8 and async (TMA/warp-specialized) execution.

---

## Key numbers

H100 SXM (Hopper) as the reference, with current data-center parts for context. Tensor Core FLOPS are dense (non-sparse); FP8/FP4 figures are the headline tensor numbers.

| Property | H100 SXM | H200 SXM | B200 (Blackwell) |
|---|---|---|---|
| FP16 Tensor TFLOPS | ~990 | ~990 | ~2,250 |
| FP8 Tensor TFLOPS | ~1,979 | ~1,979 | ~4,500 |
| FP4 Tensor TFLOPS | — | — | ~9,000 |
| HBM capacity | 80 GB HBM3 | 141 GB HBM3e | 180–192 GB HBM3e |
| HBM bandwidth | ~3.35 TB/s | ~4.8 TB/s | ~8 TB/s |
| Shared memory / SM | ~228 KB | ~228 KB | ~228 KB |
| NVLink (per GPU) | 900 GB/s (Gen4) | 900 GB/s (Gen4) | 1.8 TB/s (Gen5) |

Note: an older, widely-cited "312 TFLOPS / ~800 GB/s" pairing refers to A100-class FP16, not the H100 — keep generations straight when budgeting.

---

## The one thing to internalize

**Every op is either compute-bound or memory-bound. Measure which before you optimize**, because the fix differs:

- **Compute-bound** → reduce FLOPs: lower precision (FP8/FP4), sparsity, better algorithms.
- **Memory-bound** → reduce data movement: fuse kernels, stage through shared memory, lower precision, raise arithmetic intensity.

FlashAttention, kernel fusion, and quantization are, at bottom, all attacks on the **memory-bandwidth wall**: modern GPUs have far more compute than their memory system can feed.

---

*Back to the [concept library index](../bricks/README.md).*
