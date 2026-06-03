# Kernel Fusion

**One-liner:** Merge multiple elementwise/reduction ops into a single GPU kernel so intermediates stay in registers/SRAM instead of round-tripping through HBM — the standard cure for memory-bound work, and why `torch.compile` and fused optimizers exist.

## The key insight

Memory-bound ops (most non-matmul work) are limited by HBM bandwidth, not FLOPs. Each unfused kernel pays: launch overhead + read inputs from HBM + write outputs to HBM. Chaining `y = a*x + b; z = gelu(y)` as two kernels writes `y` to HBM and reads it back. Fusing keeps `y` in registers — one launch, inputs read once, only the final result written.

Arithmetic intensity (FLOPs per byte) rises because the byte denominator drops. On the roofline this slides the op rightward; for a long elementwise chain, speedup ≈ (number of fused HBM passes saved).

```
# unfused: 2 launches, ~5 HBM passes (read x,b; write y; read y; write z)
y = a * x + b      # kernel 1: read x,b -> write y
z = gelu(y)        # kernel 2: read y   -> write z

# fused: 1 launch, ~2 HBM passes (read x,b; write z), y never touches HBM
z = gelu(a * x + b)  # one kernel, y lives in registers
```

## Where it appears

- **torch.compile / TorchInductor** — traces the graph and codegen's fused Triton (GPU) / OpenMP (CPU) kernels for elementwise + reduction chains; the headline reason it beats eager mode on bandwidth-bound code.
- **Fused optimizers** — `fused=True` AdamW, Apex `FusedAdam`, foreach/multi-tensor ops: the whole moment-update + weight-step runs in one kernel over all params instead of dozens of tiny launches.
- **FlashAttention** — fuses QKᵀ, softmax, and the V product into one kernel so the N×N attention matrix never materializes in HBM (fusion + tiling + online softmax).
- **Fused LayerNorm/RMSNorm, bias+GeLU, residual+norm** — staples of every transformer kernel library (Triton, cuDNN, Liger-Kernel).

## Common mistake

Expecting fusion to speed up compute-bound (matmul-dominated) kernels. Fusion attacks HBM traffic and launch overhead — it helps memory-bound ops. A big GEMM is already compute-bound; fusing an epilogue onto it helps only the epilogue, not the matmul FLOPs. Profile first: if you're not bandwidth-bound, fusion buys little.

## See also
- [[roofline]] — fusion raises arithmetic intensity, moving memory-bound ops toward the compute roof
- [[flash-attention]] — fusion + tiling applied to the attention kernel
- [[triton]] — the DSL most fused kernels (and torch.compile output) are written in
