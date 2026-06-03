# FP8 / Low-Bit Inference & Fused Kernels

**One-liner:** Serve weights/activations (and increasingly attention) in FP8 or sub-byte INT with calibrated per-tensor/per-channel scales, fused into dequant-GEMM kernels so you halve memory traffic and double Tensor-Core throughput — pushing decode past the memory wall without a separate dequant pass.

## The formula / definition

Quantize-dequantize for a tensor block with scale s (and optional zero-point z):
- `q = clip(round(x / s) - z, qmin, qmax)`  →  `x̂ = s · (q + z)`
- FP8 has *two* IEEE-ish layouts: **E4M3** (4-exp/3-mant, ±448, used for weights & activations) and **E5M2** (5-exp/2-mant, ±57344, used for gradients/wide-range). No INT zero-point — the float mantissa carries it.
- Scale granularity matters: per-tensor (1 scale) → per-channel/per-row → **block/microscaling (MXFP8/MXFP4: one shared 8-bit exponent per 32-element block)**.

Fused dequant-GEMM: `Y = (s_a ⊙ A_q) @ (s_w ⊙ W_q)ᵀ` where dequant happens in the GEMM epilogue/prologue *in registers* — never materializing the full-precision matrix in HBM. INT4-weight + FP16-activation ("W4A16") kernels (Marlin, Machete, EXL3) dominate low-batch decode; W8A8-FP8 dominates high-batch prefill.

## Where it appears

- **FlashAttention-3 / DeepSeek-V3** — FP8 attention and FP8 GEMMs in the training+inference path; DeepSeek uses fine-grained block-scaled FP8 with FP32 accumulation on Hopper.
- **TensorRT-LLM / vLLM / SGLang** — FP8 KV-cache + W8A8 serving; vLLM ships Marlin/Machete W4A16 kernels for AWQ/GPTQ checkpoints.
- **NVFP4 / MXFP4 (Blackwell)** — 4-bit float block formats with hardware-accelerated micro-scaling; OpenAI's gpt-oss ships MXFP4 MoE weights.
- **SmoothQuant / AWQ / GPTQ** — produce the calibrated low-bit checkpoints these fused kernels consume.

## Common mistake

Thinking FP8 ≈ "INT8 but float, same precision." E4M3 has only ~2 decimal digits of mantissa precision and a tiny dynamic range (±448) — without per-tensor/block scaling tuned to activation magnitudes, outlier channels saturate or underflow. Also: forgetting to **accumulate in FP32**; chaining FP8 multiplies into FP8 accumulators destroys quality even when the inputs are fine.

## See also
- [[kv-cache-quantization]] — the KV cache is often the larger low-bit win during decode
- [[mxfp-microscaling-block-formats]] — the block-scaled formats that make sub-byte float viable
- [[kernel-fusion]] — why dequant must live inside the GEMM, not before it
