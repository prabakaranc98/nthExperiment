# Roofline Model

**One-liner:** Every GPU operation is either compute-bound or memory-bandwidth-bound; the roofline tells you which, and therefore how to optimize it.

## The model

Performance (FLOP/s achieved) = min(Peak FLOP/s, Bandwidth × Arithmetic Intensity)

**Arithmetic intensity** = FLOPs / Bytes transferred (to/from HBM)

- High intensity (e.g., large matmul): **compute-bound** — you're limited by FLOP/s
- Low intensity (e.g., elementwise ops, layer norm): **memory-bound** — you're limited by bandwidth

On an H100 SXM:
- Peak FP16 FLOP/s: ~312 TFLOP/s
- HBM bandwidth: ~800 GB/s
- Ridge point (crossover): 312e12 / 800e9 ≈ 390 FLOPs/byte

Operations with arithmetic intensity > 390 are compute-bound; below are memory-bound.

## Implications

| Operation | Arithmetic intensity | Bound |
|-----------|---------------------|-------|
| Large matmul (N>>1) | O(N) | Compute |
| Elementwise (ReLU, GELU) | ~1 FLOP/byte | Memory |
| Layer norm / softmax | ~4-8 FLOPs/byte | Memory |
| Standard attention (full N²) | O(N) but writes N² | Memory (due to HBM traffic) |
| FlashAttention | O(N), no N² materialization | Compute |

## How to use it

Before optimizing anything:
1. Count FLOPs for the operation
2. Count bytes of HBM traffic (reads + writes)
3. Compute arithmetic intensity
4. Compare to ridge point → know which resource to optimize

**Memory-bound fix:** kernel fusion (do multiple operations in one pass without writing intermediate results to HBM). This is what Triton fused kernels do.

**Compute-bound fix:** reduce precision (FP16 → FP8), use tensor cores, reduce FLOPs.

## Where it appears

- FlashAttention design — moving attention from memory-bound to compute-bound
- Triton kernel writing — you're solving the memory-bound problem
- MFU (Model FLOP Utilization) — measures how close to peak compute you're running

## Common mistake

Assuming that faster FLOP/s always helps. If your operation is memory-bound, a faster GPU with the same bandwidth won't help. You need more bandwidth or less data movement.

## See also
- [[flash-attention]] — the canonical example of roofline reasoning applied
- [[mixed-precision]] — moving to FP8 doubles compute capacity; only helps if compute-bound
