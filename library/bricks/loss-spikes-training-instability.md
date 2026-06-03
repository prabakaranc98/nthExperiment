# Loss Spikes & Training Instability

**One-liner:** Sudden mid-training loss divergences at scale, traced to attention-logit blowup, exploding output/activation norms, bf16 precision limits, or bad data shards — every large-model tech report documents the mitigations.

## The key insight

Spikes are growth in a positive-feedback loop that softmax/normalization usually masks until it saturates. The dominant mechanism: attention logits z = q·k/√d grow large, softmax saturates toward one-hot, gradients vanish on the saturated path while the few unsaturated entries get huge updates — the classic "attention entropy collapse." Symptoms and the standard fixes:

- **Attention logit growth** → QK-norm: normalize q,k before the dot product so logit magnitude is bounded regardless of weight growth.
- **Output logit growth** → z-loss: add `1e-4 · (log Σ_j exp(z_j))^2` to keep the softmax partition function near 1 (PaLM, Chinchilla, Baichuan).
- **Exploding update / activation norms** → gradient clipping (global-norm), lower/decayed LR, and Adam ε raised (1e-8 → 1e-15 fix or 1e-12) so tiny second-moment v doesn't make `m/(√v+ε)` explode.
- **bf16 dynamic range** → bf16 has 8 exp bits (huge range) but only 7 mantissa bits; rounding noise in `1+x` accumulations and Adam states triggers instability — fix with fp32 master weights/optimizer states and stochastic rounding.
- **Bad data shard** → a corrupted/duplicated/degenerate batch produces an outlier gradient; mitigation is "skip the batch, rewind to a pre-spike checkpoint, shuffle past the offending data, resume."

Empirically (PaLM 540B) spikes were *not* reproducible from the same step on a different shard ordering — implicating data×state interaction, not a single bad example.

## Where it appears

- **PaLM (540B)** — ~20 spikes over training; mitigated purely by rewind-to-checkpoint + skip ~200–500 batches, no architecture change.
- **GLM-130B / OPT-175B logbook** — frequent spikes; OPT switched optimizers and clipped; GLM used embedding-gradient shrink and DeepNorm.
- **Chinchilla, PaLM, Baichuan, Gemma** — z-loss as standard regularizer.
- **ViT-22B, Gemma 2, Chameleon** — QK-norm to tame attention logits, especially for multimodal/high-LR runs.
- **Tele-FLM / small-scale-to-large "μP + spike prediction"** — using small proxy runs to predict and pre-empt large-run instability.

## Common mistake

Treating a spike as a one-off RNG fluke and just lowering the global LR. The LR cut often only delays the divergence; the actual cause is usually a specific mechanism (unbounded logits, Adam ε, a data shard). Diagnose the norm/logit traces first — and remember a spike that *recovers on its own* (loss returns to trend) is benign and should not be "fixed."

## See also
- [[z-loss-logit-stabilization]] — the canonical output-logit fix
- [[qk-normalization]] — bounds attention logits at the source
- [[gradient-clipping]] — the blunt first line of defense against outlier updates
