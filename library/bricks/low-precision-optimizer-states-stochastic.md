# Low-Precision Optimizer States & Stochastic Rounding

**One-liner:** Store optimizer moments (and sometimes master weights) in 8-bit/fp8 with per-block dynamic quantization, and apply stochastic rounding to bf16/fp16 weight updates so tiny updates aren't silently truncated — cutting memory/bandwidth ~2-4x while matching fp32 convergence.

## The two ideas

**1. Block-wise quantized states (8-bit Adam, Dettmers et al. 2022).** Adam holds two fp32 states per parameter (m, v) — 8 bytes/param, more than the model itself. Instead, store each as 8-bit. Quantize per *block* (e.g. 2048 elements) so one outlier doesn't blow up the whole tensor's scale:

```
block_b: q_i = round_to_codebook( s_i / absmax(block_b) )   # 8-bit index
dequant: s_i ≈ q_i * absmax(block_b)
```

Uses a **dynamic (non-linear) quantization map** — codebook spaced to give more resolution near zero where most state values live. v can also use a quantile-uniform map. States are dequantized to fp32 only inside the update, then re-quantized. ~75% state-memory reduction vs fp32 Adam.

**2. Stochastic rounding (SR).** When applying `w ← w − lr·Δ` in low precision (bf16, 8-bit mantissa), round-to-nearest discards updates smaller than the weight's ULP, stalling training. SR rounds *up* with probability proportional to the residual:

```
w_lo = floor(w_hi to bf16 grid);  r = (w_hi − w_lo)/ulp ∈ [0,1)
w_bf16 = w_lo + ulp  if u < r  else w_lo,   u ~ Uniform[0,1)
```

This makes rounding **unbiased**: E[round(x)] = x. Lets you keep weights in bf16 with no fp32 master copy, since expected accumulation is correct.

## Where it appears

- **bitsandbytes 8-bit Adam/AdamW/LAMB** — the canonical implementation; block-wise dynamic quant of m,v. Pairs with QLoRA fine-tuning (frozen 4-bit weights + 8-bit optimizer on adapters).
- **fp8 optimizer states** — FP8-LM, Transformer Engine, ZeRO/DeepSpeed fp8: moments and/or master weights in fp8 (E4M3/E5M2) with per-tensor or block scaling.
- **Stochastic rounding** — used to drop the fp32 master copy in bf16 training (Gopher/Chinchilla-era recipes, AMD/Graphcore stacks); standard in low-bit accumulation and fp8 training pipelines.
- **8-bit optimizers + ZeRO/FSDP** — combine sharding with low-bit states to fit large models on fewer GPUs.

## Common mistake

Using round-to-nearest for the bf16/8-bit *weight update* instead of stochastic rounding. With round-to-nearest, when |lr·Δ| < ½·ulp(w) every update rounds back to the same value — the model freezes despite nonzero gradients. SR fixes this by being unbiased in expectation. Separately: don't quantize states with a single global scale — outliers force coarse resolution everywhere; block-wise scaling is what makes 8-bit states actually work.

## See also
- [[mixed-precision]] — the master-weights/loss-scaling regime that SR lets you partially eliminate
- [[adamw]] — the optimizer whose m,v states are being compressed
- [[fp8-low-precision-training]] — fp8 formats and scaling used for both states and compute
