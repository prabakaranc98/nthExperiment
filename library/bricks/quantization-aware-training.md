# Quantization-Aware Training (QAT)

**One-liner:** Insert fake-quant ops in the forward pass and backprop through them with the straight-through estimator, so weights learn to live on the low-bit grid — recovering the accuracy PTQ loses at INT4 and below, at the cost of a (fine-)training run.

## The formula / definition

Fake-quant op (simulated quantize-dequantize in fp): forward computes
  x̂ = s · (clamp(round(x/s) + z, q_min, q_max) − z)
i.e. quantize then immediately dequantize, so the network sees the rounding error but keeps fp tensors and fp matmuls during training.

The round() has zero gradient a.e., so QAT uses the **straight-through estimator (STE)**: treat the rounded op as identity on the backward pass, but gate the gradient to zero outside the representable range:
  ∂L/∂x = ∂L/∂x̂ · 1[q_min ≤ round(x/s)+z ≤ q_max]

Scales/zero-points may be fixed (from calibration) or learned. **LSQ** (Esser et al., 2020) makes s a trainable parameter, deriving ∂x̂/∂s in closed form and scaling its gradient by 1/√(N·q_max) for stable updates — the standard modern QAT recipe.

## Where it appears

- **LSQ / LSQ+ / PACT** — learn the step size (and clipping bound) jointly with weights; the canonical sub-8-bit QAT methods.
- **LLM-QAT** (Liu et al., 2023) — data-free QAT for LLMs using the model's own generations as training data; quantizes weights, activations, and the KV cache.
- **BitNet / BitNet b1.58** (Microsoft, 2024) — trains ternary {−1,0,1} weights from scratch with STE; "QAT is the only way to get good 1.58-bit," not a post-hoc fix.
- **QAT distillation pipelines** — Llama / Gemma / Qwen quantized releases use short QAT + KD passes (teacher = fp model) to ship INT4 weights with minimal accuracy drop.

## Common mistake

Reaching for QAT first. PTQ + outlier-aware tricks (GPTQ/AWQ/SmoothQuant) already gets you INT8 and usually decent INT4 with no gradients — QAT is the expensive last resort for sub-4-bit or activation/KV-cache quantization where PTQ collapses. Second mistake: forgetting that STE only simulates quantization in fp; deploying still needs a real low-bit kernel, and the trained scales/zero-points must be exported to match it exactly.

## See also
- [[quantization]] — QAT is the train-through-it branch; PTQ + GPTQ/AWQ is the cheaper post-hoc branch
- [[gumbel-softmax-straight-through-estimator]] — same STE trick for backprop through a non-differentiable discretization
- [[extreme-sub-4-bit-quantization]] — the regime (BitNet, INT2/ternary) where QAT becomes mandatory
