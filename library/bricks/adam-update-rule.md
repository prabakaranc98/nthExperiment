# Adam Update Rule (the m/v EMA equations)

**One-liner:** Per-parameter adaptive optimizer that keeps EMAs of the gradient (1st moment m) and its square (2nd moment v), bias-corrects both, and steps θ ← θ − η·m̂/(√v̂+ε) — a sign-like, scale-invariant update normalized by the gradient's RMS.

## The update equations

At step t, with gradient gₜ = ∇_θ L:

- mₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ        (1st moment EMA)
- vₜ = β₂·vₜ₋₁ + (1−β₂)·gₜ²       (2nd moment EMA, elementwise square)
- m̂ₜ = mₜ / (1 − β₁ᵗ)             (bias correction)
- v̂ₜ = vₜ / (1 − β₂ᵗ)
- θₜ = θₜ₋₁ − η · m̂ₜ / (√v̂ₜ + ε)

Defaults: β₁=0.9, β₂=0.999 (often 0.95 for LLMs), ε=1e-8. m,v initialize at 0; bias correction undoes the resulting startup underestimate. Note ε is **outside** the sqrt in the canonical form.

## Where it appears

- **Essentially all LLM pretraining** — AdamW (Adam + decoupled weight decay) is the default optimizer for GPT/Llama/etc.
- **Adafactor / 8-bit Adam / low-precision states** — compress or factorize the v (and m) buffers to cut the 2×–3× param-count memory overhead Adam imposes.
- **Muon, Shampoo/SOAP** — 2024–2026 challengers framed against Adam; often still use Adam for 1D params (embeddings, norms) while using a matrix-aware rule elsewhere.

## Common mistake

Confusing v (the **EMA of squared gradients**, a variance-like denominator) with the actual gradient variance, or thinking √v̂ acts per-step. It's a smoothed running estimate, so the effective step size ≈ η·sign(g) only when gradients are stationary; with noisy or sparse gradients the normalization lags and the update is not pure sign descent.

## See also
- [[adamw]] — the dominant variant; moves weight decay outside the m/v normalization
- [[adam-hyperparameters]] — how to set β₁, β₂, ε, η in practice
- [[muon-optimizer]] — spectral-norm alternative that often replaces Adam on 2D weights
