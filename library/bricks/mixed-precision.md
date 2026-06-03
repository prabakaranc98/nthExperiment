# Mixed Precision

**One-liner:** Run compute in 16-bit (FP16/BF16) for speed and memory while keeping an FP32 master copy of weights — and for FP16, loss scaling — to preserve numerical stability; standard since 2018, with BF16 winning because its FP32-range exponent removes the need for scaling.

## The formula / definition

Two 16-bit formats, differing in how 16 bits split into exponent/mantissa:

- **FP16** (IEEE half): 1 sign, 5 exponent, 10 mantissa. Range ≈ ±6e-5 to 65504, ~3-4 decimal digits.
- **BF16** (bfloat16): 1 sign, 8 exponent, 7 mantissa. Same dynamic range as FP32 (~1e-38 to 3e38), but coarser precision.

Standard training loop (the AMP recipe):

```
fp32_master   = weights in FP32          # the "master copy"
fp16_weights  = cast(fp32_master)        # used for the forward/backward
loss          = forward(fp16_weights, x)
scaled_loss   = loss * S                 # loss scaling (FP16 only)
scaled_grads  = backward(scaled_loss)    # grads computed in FP16
grads         = scaled_grads / S         # unscale back to true magnitude
optimizer.step(fp32_master, grads)       # update happens in FP32
```

**Why each piece:** FP16 gradients underflow to 0 below ~6e-5; multiplying the loss by S (e.g. 2^k) shifts grads up into the representable range, then you divide back before the FP32 weight update. **Dynamic loss scaling** auto-tunes S: skip the step and halve S on inf/NaN, double S after N stable steps. The FP32 master copy prevents the slow drift where small updates (w += tiny) round to no-op in 16-bit.

**BF16 needs no loss scaling** — its 8-bit exponent already spans the FP32 range, so gradients don't underflow. You trade mantissa bits (precision) for range, which training tolerates well.

## Where it appears

- **NVIDIA Apex / `torch.cuda.amp` (autocast + GradScaler)** — the original 2018 mixed-precision recipe (Micikevicius et al.); autocast picks per-op precision, GradScaler does dynamic loss scaling.
- **Every frontier LLM pretrain** — done in BF16 (GPT, Llama, etc.) once Ampere/TPU hardware made BF16 native; no GradScaler needed.
- **FP8 training (H100/Blackwell, Transformer Engine, FlashAttention-3)** — the 2024-2026 frontier: matmuls in FP8 (E4M3/E5M2) with per-tensor/per-block scaling factors, accumulation in higher precision — a generalization of the same master-copy + scaling idea.
- **Optimizer states** — Adam moments often kept FP32 even when weights are BF16, since variance estimates need the precision.

## Common mistake

Thinking BF16 is strictly "better" than FP16 — it has *less* mantissa precision (7 vs 10 bits). BF16 wins for training because range matters more than precision there. The other mistake: forgetting the FP32 master copy and accumulating the optimizer step directly in 16-bit, which silently stalls learning as small updates round away. Also: applying loss scaling under BF16 — it is unnecessary and just adds a failure mode.

## See also
- [[quantization]] — also reduces numerical precision, but for inference/storage with integer formats rather than training compute
- [[training-stability]] — loss scaling and FP32 master weights are core stability tooling
- [[roofline]] — half precision halves bytes moved, shifting memory-bound kernels along the roofline
