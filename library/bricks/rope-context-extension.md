# RoPE Context Extension (NTK / YaRN / Position Interpolation)

**One-liner:** Extend a RoPE model's usable context past its training length by rescaling rotation frequencies — linear Position Interpolation squeezes all positions, NTK-aware scaling stretches the base, and YaRN does per-frequency wavelength interpolation plus an attention-temperature correction — usually with brief fine-tuning.

## The setup

RoPE rotates query/key dim-pair i by angle m·θᵢ at position m, with θᵢ = base^{−2i/d}, base = 10000. Train length L_train; target length L_eval = s·L_train (scale factor s). Naively running at L_eval fails: high positions hit angles never seen in training → attention degenerates. All methods reduce the rotation rate so position L_eval maps onto angles the model already learned.

## The three methods

**Position Interpolation (PI, Chen et al. 2023):** rescale position m → m/s for every frequency. Equivalent to m·θᵢ → (m/s)·θᵢ. Uniform squeeze; cheap but crushes high-frequency (local) detail → needs fine-tuning, hurts short-context perplexity.

**NTK-aware (bloc97, 2023):** instead of scaling positions, scale the *base*: base' = base·s^{d/(d−2)}. High-freq dims (small i) are barely touched; low-freq dims interpolate. Distributes the "stretch" across the spectrum → works *training-free* for modest s. "Dynamic NTK" recomputes s at inference from the current sequence length.

**YaRN (Peng et al. 2023):** per-frequency ("NTK-by-parts") interpolation gated by wavelength λᵢ = 2π/θᵢ vs context size:
- short λ (high freq) → no interpolation (keep θᵢ);
- long λ (low freq) → full interpolation (θᵢ/s);
- middle → linear ramp.
Plus a length-scaling **attention temperature**: scale logits by 1/t with √(1/t) ≈ 0.1·ln(s) + 1, which sharpens the softmax to offset the entropy increase from longer sequences. Best perplexity, ~10x less fine-tuning data than PI.

## Where it appears

- **Code Llama / Llama 3.1 (128k)** — RoPE base increased (NTK-style), then long-context continued pretraining.
- **YaRN in Qwen2/2.5, Mistral, DeepSeek** — the standard recipe for 32k→128k+ extension; vLLM/HF expose `rope_scaling={"type":"yarn", ...}`.
- **Dynamic NTK** — common training-free inference hack in HF `transformers` for graceful length overflow.

## Common mistake

Applying the scale factor *uniformly to all frequencies* (plain PI / naive base scaling) and assuming no fine-tuning is needed. Uniform interpolation damages high-frequency dims that encode local/positional precision, tanking short-context quality; and forgetting YaRN's attention-temperature term loses much of its gain. Match the configured scale to the actual deployed length — over-scaling a short prompt hurts.

## See also
- [[rope]] — the rotary embedding these methods rescale
- [[long-context-eval-methodology]] — extension is worthless without testing actual retrieval, not just perplexity
- [[nope-positional-scheme-length-generalization]] — the alternative: remove explicit positions to generalize length
