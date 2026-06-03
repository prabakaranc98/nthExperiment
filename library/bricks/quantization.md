# Quantization

**One-liner:** Store/compute weights and activations in low-bit formats (INT8/INT4/FP8) to cut memory and boost throughput; PTQ (cheap, post-hoc) vs QAT (trains through fake-quant), with outlier-aware schemes (GPTQ/AWQ/SmoothQuant) to survive 4-bit.

## The formula / definition

Affine (asymmetric) quantization of a tensor x to b bits:

  scale s = (max − min) / (2^b − 1),  zero-point z = round(−min / s)
  q = clamp(round(x / s) + z,  0,  2^b − 1)
  x̂ = s · (q − z)   ← dequantized value

Symmetric variant (z = 0, range [−2^{b−1}, 2^{b−1}−1]) is standard for weights. Granularity matters: per-tensor < per-channel (per output dim) < per-group (e.g. groups of 128) — finer granularity = more scales stored, less error.

## PTQ vs QAT

- **PTQ** (post-training): quantize a trained model with a small calibration set to fit scales. Cheap, no gradients. Good to INT8; needs tricks below for INT4.
- **QAT** (quantization-aware training): insert fake-quant ops in the forward pass, train through them with the straight-through estimator (STE: gradient passes through round() as identity). Recovers accuracy at low bit-width but costs a training run.

## Outlier handling (why naive INT4 fails)

LLM activations have a few channels with huge magnitudes; one outlier blows up the per-tensor scale and crushes everyone else's resolution.
- **GPTQ** — layer-wise weight quant minimizing ‖WX − ŴX‖² via second-order (Hessian) error compensation; 3-4 bit weights.
- **AWQ** — protect the ~1% salient weight channels (chosen by activation magnitude) by per-channel scaling before quant; weight-only INT4.
- **SmoothQuant** — migrate activation outliers into the weights via a per-channel scale so both become quantizable (enables W8A8).
- **FP8 (E4M3/E5M2)** — native on H100/Blackwell; floating layout tolerates outliers far better than INT8 and is now standard for training and inference.

## Where it appears

- llama.cpp / GGUF (Q4_K_M etc.) and bitsandbytes NF4 — run 70B+ on consumer GPUs
- QLoRA — frozen 4-bit (NF4) base + LoRA adapters in bf16; fine-tune on one GPU
- vLLM / TensorRT-LLM — FP8 and INT4 weight-only kernels for serving throughput
- DeepSeek-V3, frontier training — FP8 mixed-precision training at scale

## Common mistake

Conflating *storage* bit-width with *compute* bit-width. Weight-only INT4 (GGUF, AWQ, QLoRA) dequantizes to fp16/bf16 before the matmul — it saves memory/bandwidth (memory-bound decode) but does NOT use INT4 tensor cores. True low-bit speedup needs W8A8/FP8 where activations are quantized too and the matmul runs in the low-precision unit.

## See also
- [[lora]] — QLoRA combines 4-bit base weights with LoRA adapters
- [[mixed-precision]] — FP8/bf16 training is the compute-side sibling of quantization
- [[roofline]] — weight-only quant helps because decode is memory-bandwidth-bound
