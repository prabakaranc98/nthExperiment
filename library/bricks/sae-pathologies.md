# SAE Pathologies (Shrinkage, Dead Features, Absorption, Splitting)

**One-liner:** The four systematic failure modes of sparse-autoencoder training — L1 shrinks active magnitudes, latents die and never fire, broad features get absorbed into co-firing specific ones, and widening the dictionary fragments one concept into many — each a learned artifact of the recon+sparsity objective rather than real model structure.

## The formula / definition

Recall the vanilla objective on activations x ∈ ℝ^d with overcomplete latent z ∈ ℝ^m (m ≫ d):

L = ‖x − x̂‖² + λ‖z‖₁,   z = ReLU(W_enc(x − b_dec) + b_enc),   x̂ = W_dec z + b_dec

**Shrinkage** — the L1 term penalizes magnitude, so the optimal z systematically *under*estimates the true feature activation even when the support (which features fire) is correct. For a single feature with true coefficient a*, the L1-optimal estimate is â = a* − λ/2 (soft-thresholding bias), so reconstructions are attenuated. Gated/JumpReLU/TopK exist to decouple *gating* (does it fire) from *magnitude* (how much).

**Dead features** — latents whose pre-activation stays below threshold for the whole training distribution; gradient is ~0, so they never recover. Often 10–90% of latents at large width. Mitigations: resampling/reinitialization (Anthropic), auxiliary "ghost grad"/AuxK loss reviving the top-k dead latents (OpenAI TopK).

**Absorption** — a general feature (e.g. "starts with S") gets partially merged into a more specific co-occurring feature (e.g. "Sydney"): the specific latent steals the general direction's contribution whenever they co-fire, so the general latent spuriously *fails* to activate on those tokens. A sparsity-driven artifact: encoding the shared component once is cheaper under L0/L1.

**Splitting** — increasing width (expansion factor) fractures one coarse feature into many finer geometrically-clustered variants; not strictly a bug (genuine finer concepts) but confounds "feature count" metrics and cross-width comparison.

## Where it appears

- **Gated SAE / JumpReLU (DeepMind, 2024)** — explicitly motivated as the fix for L1 shrinkage; learnable threshold gates firing without an L1 magnitude tax.
- **TopK / BatchTopK SAE (OpenAI, 2024)** — exact-L0 gating sidesteps shrinkage; ships AuxK loss specifically to resurrect dead latents.
- **Feature absorption studies (Chanin et al. 2024; "A is for Absorption")** — formalize + measure absorption on first-letter/spelling probes as a failure of SAE faithfulness.
- **Gemma Scope / SAEBench (2024–2025)** — report dead-latent fraction, sparsity-fidelity frontier, and absorption as standard SAE quality axes.

## Common mistake

Reading these as orthogonal "tune-it-away" knobs. They share one root cause — the sparsity penalty trades faithfulness for compression — so fixing one can worsen another (e.g. higher width cuts reconstruction loss but increases splitting and absorption). Low reconstruction loss is NOT evidence of clean, monosemantic, causally-faithful features.

## See also
- [[sparse-autoencoders]] — the base method whose objective produces these artifacts
- [[sae-variants]] — Gated/JumpReLU/TopK are direct engineered responses to shrinkage and dead latents
- [[superposition]] — absorption/splitting are how SAEs mis-resolve features the model packed into shared directions
