# AWQ (Activation-Aware Weight Quantization)

**One-liner:** Weight-only PTQ that protects the salient ~1% of weight channels — identified by *activation* magnitude, not weight magnitude — via a per-channel scaling search, with no backprop or reconstruction; the de facto W4A16 method shipped in most inference engines.

## The key insight

Quantization error is dominated by a few salient weight channels. Which channels are salient is determined by the **activation** that multiplies them, not by the weight values. Keeping the top 1% in FP16 (mixed precision) fixes accuracy but is hardware-ugly; AWQ instead achieves the same protection with a uniform low-bit scheme by *scaling*.

For weight column w and activation x, the product is invariant under w·x = (w·s)·(x/s). Scaling up a salient channel by s before quantizing shrinks its **relative** quant error by ~1/s, while x/s is folded into the previous layer (LayerNorm / preceding linear) at no inference cost.

Per output-channel (or per group) scale: minimize quant MSE
  s* = argmin_s || (w · s) · Q(w·s)^{-1} ... ||  — in practice a **grid search** over s = (mean|x|)^α, α ∈ [0,1], picking α that minimizes the layer output error on a tiny calibration set.

Quantize is standard group-wise round-to-nearest (e.g. group size 128, INT4):
  Q(w) = Δ · round(w/Δ + z),   Δ = (max−min)/(2^b−1)

No gradients, no per-weight reconstruction (unlike GPTQ's Hessian-based update) — just a closed activation pass + scale search. Fast (minutes) and robust under low calibration data / distribution shift.

## Where it appears

- **AWQ paper** (Lin et al., MLSys 2024) + the **TinyChat / llm-awq** kernels — original W4A16 with fused dequant.
- **vLLM, TensorRT-LLM, llama.cpp, MLC-LLM, HF Transformers** — `awq` is a standard checkpoint format for 4-bit serving on consumer/edge GPUs.
- Default 4-bit quant path for many community Llama / Qwen / Mistral releases ("...-AWQ" on the Hub).

## Common mistake

Thinking AWQ keeps salient weights in FP16 — that's the *mixed-precision baseline it rejects*. AWQ quantizes **everything** to the same bit-width; the salient channels are protected purely by the activation-aware per-channel **scale**, which is then absorbed into adjacent ops so inference stays uniform-precision. Also: AWQ is **weight-only (W4A16)** — activations stay FP16, so it does not fix activation outliers the way SmoothQuant does for W8A8.

## See also
- [[gptq]] — the other dominant 4-bit weight-only PTQ; Hessian/reconstruction-based vs AWQ's scale search
- [[activation-outliers-smoothquant]] — shares the "migrate difficulty via per-channel scaling" trick, but for W8A8 activation quant
- [[quantization]] — the umbrella: bit-widths, group size, RTN vs learned
