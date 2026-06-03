# Activation Outliers / SmoothQuant

**One-liner:** A handful of feature channels in LLM activations grow to 10-100x the typical magnitude (concentrated, persistent, emerging at ~6.7B params), wrecking per-tensor activation quantization; SmoothQuant migrates that difficulty from activations into weights via a per-channel scale so both can be quantized cheaply (W8A8), while LLM.int8() instead keeps the outlier dimensions in FP16.

## The phenomenon

For a layer input X (tokens × C_in), a small set of channels c* have huge magnitude that recurs across tokens and is roughly fixed across inputs. A single per-tensor activation scale s = max|X| / (2^{b-1}-1) is dominated by these channels, so everyone else gets ~0-1 quantization levels → catastrophic error. Outliers live in *channels* (columns), not random entries — that's what makes per-channel migration possible. Weights, by contrast, are flat/Gaussian and easy to quantize; the difficulty is asymmetric.

## SmoothQuant: migrate the difficulty

Y = (X · diag(s)^{-1}) · (diag(s) · W) = X̂ · Ŵ, mathematically identical. Choose per-input-channel scale

  s_j = max_t|X_{t,j}|^α / max_i|W_{j,i}|^{1-α},  α ∈ [0,1] (typ. 0.5)

This divides outlier activation channels down and multiplies the corresponding weight rows up, balancing the dynamic range so both are now per-tensor INT8-friendly. s is folded into the *preceding* LayerNorm/Linear at no runtime cost (it's a static reparameterization, calibrated on a small set). Result: W8A8 with near-FP16 accuracy.

## LLM.int8() (the alternative)

Mixed-precision decomposition: detect outlier feature dimensions (threshold ~6.0), run those columns of X and rows of W in FP16, the rest in vectorized INT8 (per-row/per-channel), then sum. No reparameterization, but the FP16 path limits speedup and needs scatter/gather.

## Where it appears

- **SmoothQuant (Xiao et al., 2022/23)** — enables W8A8 PTQ on OPT/BLOOM/Llama with little loss; precursor to many serving INT8 paths.
- **LLM.int8() / bitsandbytes** — the original outlier diagnosis; default 8-bit inference in HF Transformers.
- **AWQ** — same outlier-channel insight but protects *weight* channels chosen by activation magnitude (weight-only INT4); contrast with SmoothQuant's activation focus.
- **Hadamard/rotation quant (QuaRot, SpinQuant)** — rotate the basis to *spread* outliers across channels instead of migrating them; now the dominant approach for 4-bit activations.

## Common mistake

Thinking SmoothQuant *removes* the outliers. It doesn't — it relocates the dynamic-range difficulty into the weights via an algebraically exact rescale; total information is conserved, you've just rebalanced what each tensor must represent. Also: α is a per-layer knob, not a universal constant; too large over-loads the weights and hurts as much as untreated activations.

## See also
- [[quantization]] — the parent framework; SmoothQuant is its outlier-handling step for W8A8
- [[awq]] — same channel-outlier observation applied weight-side for INT4
- [[hadamard-rotation-based-quantization]] — the rotate-don't-migrate successor for low-bit activations
