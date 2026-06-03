# SAE Variants (TopK / JumpReLU / Gated)

**One-liner:** Architectural fixes to vanilla L1 sparse autoencoders — TopK enforces exact k-sparsity, JumpReLU learns a per-feature activation threshold, Gated decouples which features fire from how much — all to kill L1's activation shrinkage and push the sparsity/reconstruction-fidelity Pareto frontier outward.

## The formulas

Encode pre-activations z = W_enc(x − b_dec) + b_enc, decode x̂ = W_dec·f + b_dec.

- **Vanilla (L1):** f = ReLU(z); loss = ‖x − x̂‖² + λ‖f‖₁. The L1 penalty biases all active magnitudes toward zero → **shrinkage** (systematic underestimation).
- **TopK** (Gao et al., OpenAI 2024): f = TopK(z) — keep the k largest pre-acts, zero the rest. No L1 term, so no shrinkage. k directly sets sparsity. Add an AuxK loss (reconstruct residual from dead latents) to revive dead features.
- **JumpReLU** (Rajamanoharan et al., DeepMind 2024): f_i = z_i · H(z_i − θ_i), a learned per-feature threshold θ. Loss = ‖x−x̂‖² + λ‖f‖₀, where ‖·‖₀ counts nonzeros. The Heaviside/L0 are non-differentiable, so θ is trained via **straight-through estimators** with a kernel-density pseudo-gradient.
- **Gated** (Rajamanoharan et al., DeepMind 2024): split the encoder into a gate (binary, π_gate = 1[z_gate>0]) and a magnitude path; f = π_gate ⊙ ReLU(z_mag). L1 is applied only to the gate's ReLU(z_gate), with a weight-tied auxiliary reconstruction so the penalty no longer shrinks the magnitudes.

## Where it appears

- **OpenAI GPT-4 SAEs** — TopK SAEs scaled to 16M latents; clean sparsity control and the standard high-throughput recipe.
- **Gemma Scope** (DeepMind 2024) — JumpReLU SAEs trained on every layer of Gemma 2, the largest open suite of interpretability SAEs.
- **Anthropic** — uses these on the L0-vs-loss frontier; later moves toward transcoders/crosscoders for circuit-level analysis.
- **BatchTopK / Matryoshka SAEs** (2024–2025) — extend TopK by sharing the k budget across a batch or nesting dictionaries at multiple sparsity levels.

## Common mistake

Comparing variants at "the same L1 λ" or treating the count of nonzeros as the only axis. The right comparison is the **L0-vs-reconstruction-loss (or downstream-CE) Pareto frontier** at matched dictionary size. Also: TopK's hard k means a feature that *should* fire on a simple input can get crowded out by k unrelated weakly-active features — JumpReLU/Gated allow variable per-token L0 and avoid this.

## See also
- [[sparse-autoencoders]] — the base method these variants all refine
- [[sae-pathologies]] — dead latents, feature absorption, shrinkage these aim to fix
- [[gumbel-softmax-straight-through-estimator]] — STE machinery powering JumpReLU's L0/threshold training
