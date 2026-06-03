# Sparse Autoencoders (SAEs)

**One-liner:** Learn an overcomplete, sparsely-activating dictionary over model activations to decompose superposed features into (approximately) monosemantic directions — the workhorse of mechanistic interpretability.

## The formula / definition

Given activations x ∈ ℝ^d (e.g. a residual-stream or MLP vector), learn an encoder/decoder with a wide latent of size m ≫ d (overcomplete, expansion factor 8–64×):

z = σ(W_enc(x − b_dec) + b_enc)      # sparse codes, z ∈ ℝ^m
x̂ = W_dec z + b_dec                   # reconstruction as a sum of dictionary atoms

The columns of W_dec are the **feature directions** (the dictionary); each active zᵢ is that feature's firing strength. Train on the reconstruction-plus-sparsity objective:

L = ‖x − x̂‖² + λ ‖z‖₁

The L1 penalty is the relaxation of the true L0 (count of active features) objective. Decoder columns are unit-normed so L1 can't be gamed by shrinking weights.

## Sparsity-mechanism variants (2024–2026)

- **L1 / vanilla** (Anthropic "Towards Monosemanticity", 2023) — simple, but L1 causes systematic activation **shrinkage** (biases magnitudes down).
- **Gated SAE** (DeepMind, 2024) — separate "which features fire" from "how much," fixing shrinkage.
- **TopK / BatchTopK** (OpenAI "Scaling and evaluating SAEs", 2024) — keep the K largest zᵢ, exact L0 control, no λ to tune.
- **JumpReLU** (DeepMind, 2024) — learnable threshold; current strong default on the sparsity/fidelity frontier.

## Where it appears

- **Anthropic — Scaling Monosemanticity (2024)** — 34M-feature SAE on Claude 3 Sonnet; found steerable concepts ("Golden Gate Bridge," deception, sycophancy).
- **Gemma Scope (DeepMind, 2024)** — open suite of JumpReLU SAEs on every layer of Gemma 2; the standard public research substrate.
- **Steering & circuits** — clamp a feature to edit behavior; **transcoders / crosscoders** extend SAEs to MLP input→output and cross-layer/cross-model feature matching.

## Common mistake

Treating recovered features as *the* ground-truth units of computation. SAEs optimize reconstruction + sparsity, not interpretability — they suffer **feature splitting/absorption**, are sensitive to width and dead latents, and high reconstruction loss does not imply faithful causal features. Reconstruction fidelity ≠ the model actually uses those directions; validate with interventions/probing.

## See also
- [[matrix-rank]] — superposition packs > d features into d dims; the dictionary is overcomplete by design
- [[svd]] — contrast with linear/orthogonal decompositions SAEs deliberately go beyond
- [[alignment]] — interpreting and steering features for monitoring and control
