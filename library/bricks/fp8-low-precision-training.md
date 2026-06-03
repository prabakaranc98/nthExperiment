# FP8 / Low-Precision Training

**One-liner:** Train with 8-bit floats — E4M3 for forward/weights/activations, E5M2 for gradients — using per-tensor or fine-grained microscaling factors to keep values in range, halving memory/bandwidth and ~2x-ing tensor-core throughput over BF16 (DeepSeek-V3, Blackwell).

## The formula / definition

Two FP8 layouts trade exponent (range) vs mantissa (precision) within 8 bits (OCP FP8 spec):

- **E4M3**: 1 sign, 4 exponent, 3 mantissa. Max ≈ ±448, finer precision. Used for **forward** tensors (weights, activations) where precision matters.
- **E5M2**: 1 sign, 5 exponent, 2 mantissa. Max ≈ ±57344, wider range. Used for **gradients** (backward), which span a larger dynamic range and need the headroom.

Both formats have tiny range vs BF16, so every matmul operand is scaled by a factor before casting and unscaled after:

```
x_fp8 = cast_e4m3(x * s)            # s chosen so |x*s| fills the E4M3 range
y     = matmul(x_fp8, w_fp8)        # tensor core: FP8 in, FP32 accumulate
out   = y / (s_x * s_w)             # unscale; accumulation stays in FP32/BF16
```

**Scaling granularity** is the whole game:
- **Per-tensor** (one s per tensor, e.g. Transformer Engine "delayed scaling" using a history of recent absmax) — cheap, but one outlier wastes the range.
- **Fine-grained / microscaling (MXFP)** — share one scale per small block (e.g. 1x128 activation tiles, 128x128 weight blocks in DeepSeek-V3; MXFP8 = 32-element blocks with an E8M0 power-of-two scale). Confines outlier damage to one block.

Accumulation is **never** in FP8 — partial sums accumulate in FP32 (or BF16); only the matmul inputs are FP8.

## Where it appears

- **DeepSeek-V3 (2024)** — FP8 for most GEMMs with tile/block-wise scaling, plus increased-precision accumulation; key to its low training cost. A handful of sensitive ops (embedding, output head, norms, attention softmax) stay BF16.
- **NVIDIA Transformer Engine (H100/H200)** — the reference FP8 training library; per-tensor delayed scaling with amax history.
- **Blackwell (B200/GB200)** — native MXFP8/MXFP6/MXFP4 microscaling tensor cores; hardware-level block scales make fine-grained FP8 (and lower) the default frontier path.
- **FlashAttention-3** — FP8 attention on Hopper.

## Common mistake

Thinking the speedup is "free" once you flip to FP8. The gain comes only when the **matmul inputs are both FP8** so the FP8 tensor core fires; if you cast back to BF16 before the GEMM you get nothing. Equally common: using one E4M3 for everything — gradients need E5M2's range, and a single per-tensor scale lets one activation outlier crush precision, which is exactly why fine-grained/MX scaling exists. Sensitive ops kept in BF16 (norms, softmax, master weights, optimizer states) are not optional.

## See also
- [[mixed-precision]] — FP8 is the next rung below BF16; same master-copy + scaling idea, finer-grained
- [[mxfp-microscaling-block-formats]] — the block-scaling format that makes fine-grained FP8 robust to outliers
- [[low-precision-optimizer-states-stochastic]] — pairing FP8 compute with 8-bit moments needs stochastic rounding to avoid stalled updates
