# RMSNorm

**One-liner:** Normalize each activation vector by its root-mean-square only — no mean subtraction, usually no bias, learned per-channel scale gain g — a cheaper LayerNorm variant that became the default LM normalizer.

## The formula / definition

For input vector x ∈ ℝᵈ:

RMSNorm(x)ᵢ = (xᵢ / RMS(x)) · gᵢ,   where  RMS(x) = √( (1/d) Σⱼ xⱼ² + ε )

Contrast with LayerNorm, which also re-centers and adds a bias:

LayerNorm(x)ᵢ = ((xᵢ − μ) / √(σ² + ε)) · gᵢ + bᵢ,   μ = mean(x),  σ² = var(x)

RMSNorm = LayerNorm with μ ≡ 0 and bias dropped. Only one statistic (mean-square) instead of two (mean + variance); the ε sits inside the sqrt. Cost saving is mostly fewer reductions and fewer params, not FLOPs.

## The claimed insight (Zhang & Sennrich, 2019)

LayerNorm's benefit is re-scaling invariance, not re-centering. Dropping the mean-subtraction term loses almost nothing in quality while removing a reduction and the bias parameters — yielding a simpler, faster, equally stable normalizer.

## Where it appears

- **LLaMA / Llama 2/3, Mistral, Qwen, Gemma** — pre-norm RMSNorm is the de facto standard transformer normalizer (2023+).
- **T5 (Raffel et al., 2020)** — early large-scale adopter ("T5 LayerNorm" = RMSNorm with no bias).
- **Gemma** — uses RMSNorm with a (1 + g) gain parameterization so weights initialize at zero.
- **QK-norm** — RMSNorm applied to query/key vectors before attention to stabilize large-model training.

## Common mistake

Assuming RMSNorm is mean-free at the activation level — it is not. It does not center x; it only divides by RMS. So if x has a large nonzero mean, that DC component survives (just rescaled). It removes the *re-centering operation*, not the mean of the data.

## See also
- [[layer-norm]] — RMSNorm is LayerNorm minus mean-subtraction and bias
- [[batch-norm]] — normalizes across the batch dim instead of the feature dim
- [[training-stability]] — pre-norm RMSNorm + QK-norm are core tricks for stable large-LM training
