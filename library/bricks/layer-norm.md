# Layer Normalization

**One-liner:** Normalize each token's activation vector over the feature dimension to zero mean / unit variance, then apply a learned per-feature scale γ and shift β — batch-independent, which is why transformers use it.

## The formula / definition

For an input vector x ∈ ℝ^d (one token's hidden state), normalize across the d features:

    μ = (1/d) Σ_i x_i
    σ² = (1/d) Σ_i (x_i − μ)²
    LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β

γ, β ∈ ℝ^d are learned. Statistics are computed per-sample over features, so behavior is identical at train and inference and independent of batch size — no running averages, unlike BatchNorm.

## Where it appears

- Original Transformer (Vaswani 2017) — Post-LN: LN applied after the residual add, `x + Sublayer(x)` then LN.
- GPT-2 / GPT-3 and nearly all modern decoders — Pre-LN: `x + Sublayer(LN(x))`. Far more stable to train deep; doesn't need learning-rate warmup as desperately.
- Vision Transformers, Whisper, AlphaFold — same Pre-LN block structure ported across modalities.
- Modern variants drop the mean-centering and β → RMSNorm (LLaMA, Mistral, Gemma, most 2023+ LLMs), which is cheaper and works as well.

## Common mistake

Confusing the normalization axis with BatchNorm. LayerNorm normalizes *across features within one sample*; BatchNorm normalizes *across the batch for one feature*. This is exactly why LN suits transformers: it has no batch dependence, no train/inference mismatch, and works with variable sequence lengths and batch size 1.

## See also
- [[rmsnorm]] — drops mean-subtraction and β; the dominant variant in 2024-26 LLMs
- [[batch-norm]] — normalizes over the batch axis instead; the contrast that explains LN's design
- [[training-stability]] — Pre-LN vs Post-LN placement is a core determinant of deep-transformer trainability
