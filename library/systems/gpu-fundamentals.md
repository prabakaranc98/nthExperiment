# GPU Fundamentals

*Why GPUs are fast, why memory is the bottleneck, and how to think about performance.*

---

## Why GPUs for deep learning?

A CPU has ~16–64 cores optimized for complex, sequential tasks. A GPU has thousands of simpler cores optimized for doing the same operation on lots of data simultaneously — perfect for matrix multiplications.

**The key operation:** matrix multiply (GEMM). Attention is a GEMM. FFN is two GEMMs. Almost all compute in a transformer is GEMMs.

---

## The memory hierarchy

From fastest/smallest to slowest/largest:

```
Registers (per thread, < 1KB)     ← fastest (~1 cycle)
Shared memory / L1 cache (~100KB) ← fast (~30 cycles)  
L2 cache (~50MB)                  ← slower
HBM (GPU RAM, ~80GB on H100)      ← slow (~800GB/s bandwidth)
CPU RAM                           ← very slow
Disk / NVMe                       ← slowest
```

**The key insight:** H100 has ~312 TFLOPS of compute but only ~800 GB/s of memory bandwidth. At FP16:
- Compute speed: 312 × 10¹² FLOPs/s
- Memory bandwidth: 800 × 10⁹ bytes/s = can load 400 × 10⁹ FP16 numbers/s

For a matrix multiply of size N×N:
- Compute: O(N³) operations
- Memory reads: O(N²) elements

For large N, compute >> memory reads → **compute-bound**.
For element-wise operations (activate, normalize):
- Compute: O(N) operations  
- Memory reads: O(N) elements

→ **Memory-bound** — limited by how fast you can read/write memory.

---

## The roofline model

The **arithmetic intensity** (FLOPs / bytes) tells you whether an operation is compute-bound or memory-bound:

```
Performance = min(Peak FLOPS, Bandwidth × Arithmetic Intensity)
```

- High arithmetic intensity (e.g., large matrix multiply) → compute-bound → use all the FLOPS
- Low arithmetic intensity (e.g., element-wise ops) → memory-bound → use all the bandwidth

**The roofline:** plot performance vs. arithmetic intensity. Ops below the bandwidth line are memory-bound; ops above are compute-bound.

---

## CUDA programming model (basics)

CUDA code runs on a **grid** of **blocks** of **threads**:
- **Thread**: smallest unit, executes one instance of the kernel
- **Block**: group of threads that share **shared memory** and can synchronize
- **Grid**: all blocks in a kernel launch

Threads in a warp (32 threads) execute in lockstep — **SIMT (Single Instruction, Multiple Thread)**. If threads in a warp take different branches, the GPU must execute both branches sequentially with inactive threads masked → **warp divergence** = performance killer.

**Shared memory** is a programmer-managed L1 cache. Load data from HBM into shared memory once, do all operations there, write back once. This is how FlashAttention achieves its speedup.

---

## Why FlashAttention matters

Standard attention:
1. Compute S = QKᵀ (write S to HBM: N² floats)
2. Compute P = softmax(S) (read S from HBM, write P to HBM: 2×N² reads/writes)
3. Compute O = PV (read P from HBM: N² reads)

For sequence length N, this requires O(N²) HBM memory and O(N²) reads/writes. For N=128K, this is huge.

**FlashAttention:** fuse all three operations into one kernel using tiling — keep the intermediate matrices in on-chip shared memory and never write them to HBM. This is **IO-aware** computation.

Result: same output, O(N) HBM memory, ~2–4× speedup. The math is identical; only the memory access pattern changed.

---

## Key numbers (H100 SXM)

| Property | Value |
|----------|-------|
| FP16 FLOPS | ~312 TFLOPS |
| FP8 FLOPS | ~624 TFLOPS (2× with FP8) |
| HBM bandwidth | ~800 GB/s |
| HBM capacity | 80 GB |
| Shared memory | ~228 KB per SM |
| CUDA cores | ~16,896 |
| Interconnect (NVLink) | ~900 GB/s (H100 SXM) |

---

## The most important thing to internalize

**Everything is either compute-bound or memory-bound.** Before optimizing anything, measure which one it is. The solution is different:
- Compute-bound: reduce FLOPs (quantization, sparse computation)
- Memory-bound: reduce data movement (fuse operations, use shared memory, reduce precision)

FlashAttention, kernel fusion, and quantization are all solving the **memory bandwidth** problem. The GPU has more compute than it can feed from memory.
