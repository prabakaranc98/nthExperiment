# Pre-Norm vs Post-Norm vs Sandwich Norm

**One-liner:** Where the normalizer sits relative to the residual add — inside the branch (Pre-Norm, identity-preserving residual stream, trains deep without warmup), after the add (Post-Norm, better-conditioned features but the residual path is rescaled every layer so it needs warmup + careful init), or wrapping the sublayer on both sides (Sandwich) — the central depth-stability lever.

## The three placements

For sublayer F (attention or MLP), residual input x:

- **Post-Norm** (original Transformer, BERT): `x ← Norm(x + F(x))`
- **Pre-Norm** (GPT-2 onward, most modern LLMs): `x ← x + F(Norm(x))`
- **Sandwich** (CogView, Gopher-style): `x ← x + Norm₂(F(Norm₁(x)))` — pre-norm the input, post-norm the branch output before adding.

Pre-Norm keeps a clean identity path: the residual stream is never renormalized, so gradients flow straight to the input → stable at 100+ layers, no warmup strictly required. Cost: the residual stream variance grows ~linearly with depth, so deep Pre-Norm collapses sublayer contributions ("representation collapse" / over-smoothing) and the final output is under-normalized (fix: a final Norm before the head). Post-Norm gives each layer a well-conditioned, unit-scale input and stronger feature mixing, but every Norm rescales the residual, amplifying gradients with depth → needs LR warmup and shrinking init.

## Where it appears

- **GPT-2/3, LLaMA, most decoder LLMs** — Pre-Norm (RMSNorm variant) for train-without-warmup stability at depth.
- **DeepNorm (Wang et al. 2022)** — Post-Norm made stable to 1000 layers via `x ← Norm(α·x + F(x))` with up-weighted residual α and down-scaled init β.
- **Gemma 2 / OLMo 2** — "both-norm" / Sandwich-style: norm before AND after each sublayer to tame activation/logit growth in deep nets.
- **muP & initialization-residual-depth-scaling** — recast the choice as a residual-branch scaling problem (1/√L or 1/L branch weights) rather than purely placement.

## Common mistake

Believing Pre-Norm is strictly better because it "trains without warmup." It trades stability for capacity: the unnormalized growing residual stream means deeper Pre-Norm layers contribute progressively less, and well-tuned Post-Norm (DeepNorm) or Sandwich often reaches higher final quality. Also: forgetting the mandatory final LayerNorm/RMSNorm before the unembedding in Pre-Norm stacks.

## See also
- [[layer-norm]] — the normalizer being placed; defines the operation
- [[rmsnorm]] — the modern drop-in used in the Pre-Norm position of most LLMs
- [[initialization-residual-depth-scaling]] — branch-scaling view that makes either placement stable at depth
