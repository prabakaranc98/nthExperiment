# GPU Memory Hierarchy (HBM / SRAM / Registers)

**One-liner:** A tiered memory stack — registers, L1/shared SRAM, L2, then HBM — where each level up is ~10-100x faster but ~100-1000x smaller, so data movement (not FLOPs) usually bottlenecks ML kernels; this asymmetry is the physical reason FlashAttention and kernel fusion exist.

## The hierarchy (H100 SXM, ballpark numbers)

| Level | Size | Bandwidth | Latency |
|---|---|---|---|
| Registers | ~256 KB / SM (64K x 32-bit) | ~100s of TB/s | ~1 cycle |
| L1 / Shared SRAM | ~256 KB / SM (configurable split) | ~20-30 TB/s on-chip | ~30 cycles |
| L2 cache | ~50 MB (chip-wide) | ~5-10 TB/s | ~200 cycles |
| HBM3 (global / DRAM) | 80 GB | ~3.35 TB/s | ~400-800 cycles |

Key ratio: H100 does ~990 TFLOP/s (BF16) but only ~3.35 TB/s HBM. Arithmetic intensity break-even ≈ 990e12 / 3.35e12 ≈ ~300 FLOPs per byte read from HBM. Below that, you are **memory-bound** — the SMs idle waiting on HBM.

## The key insight

`time ≈ max(compute_time, memory_time)`, where `compute_time = FLOPs / peak_FLOP/s` and `memory_time = bytes_moved / HBM_bandwidth`. Most ML primitives (elementwise ops, softmax, layernorm, attention at inference, GEMV in decode) are memory-bound: tiny arithmetic intensity, so you pay for moving tensors to/from HBM. The fix is to **keep data in SRAM/registers and do more compute per byte** — tile + fuse so you read HBM once instead of materializing intermediates.

## Where it appears

- **FlashAttention (1/2/3)** — tiles Q,K,V into SRAM, fuses the whole attention into one kernel; never writes the N×N score matrix to HBM, cutting HBM IO from O(N²) to O(N).
- **Kernel fusion / Triton / torch.compile** — fuse elementwise + reductions (e.g. bias+GELU+dropout, fused layernorm) to avoid round-tripping activations through HBM between ops.
- **Decode-phase LLM inference** — GEMV against the KV cache is purely HBM-bandwidth-bound; this is why decode latency scales with KV-cache size, motivating GQA, MLA, and KV-cache quantization.
- **Tensor/tile MMA cores** — operate on tiles staged in shared SRAM; double-buffering hides HBM latency behind compute.

## Common mistake

Optimizing for FLOPs / counting MACs when the kernel is memory-bound. Halving the math does nothing if you still move the same bytes through HBM — the roofline ceiling is the bandwidth, not the FLOP rate. Profile arithmetic intensity first; only compute-bound kernels benefit from lower-precision math or fewer ops.

## See also
- [[roofline]] — the model that says whether a kernel is memory- or compute-bound
- [[flash-attention]] — the canonical IO-aware kernel built on this hierarchy
- [[kernel-fusion]] — eliminating HBM round-trips by keeping intermediates on-chip
