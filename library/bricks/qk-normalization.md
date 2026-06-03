# QK-Normalization

**One-liner:** Apply a normalization (typically RMSNorm/L2) to queries and keys *before* the dot product so attention logits stay bounded — preventing the logit-magnitude blowup that drives loss spikes and attention-entropy collapse at scale.

## The formula / definition

Standard scaled dot-product attention:

    logits = (Q Kᵀ) / √d_head

QK-norm inserts a per-head normalization on Q and K first:

    Q̂ = γ_q · RMSNorm(Q)        K̂ = γ_k · RMSNorm(K)
    logits = (Q̂ K̂ᵀ) / √d_head

Two common variants:
- **L2 / cosine form** (Dehghani et al., ViT-22B): Q̂ = Q/‖Q‖, K̂ = K/‖K‖, so logits = cos-sim, then scaled by a *learnable* temperature τ (≈ logit-cap by construction).
- **RMSNorm form** (most LLMs): LayerNorm/RMSNorm with learnable gain along d_head, applied after the head split (so each head normalized independently).

Effect: ‖Q̂‖, ‖K̂‖ become roughly scale-invariant, so logit growth is decoupled from activation-norm growth during training. No raw logit can run away as weights/activations grow.

## Where it appears

- **ViT-22B (Dehghani et al., 2023)** — the origin story: training diverged because attention logits grew until softmax saturated into near-one-hot (entropy collapse); QK-norm fixed it.
- **Chameleon / Gemma 2 / many 2024–25 LLMs** — QK-norm as standard stabilizer, often paired with z-loss and logit soft-capping; lets you push LR higher.
- **OLMo 2, DeepSeek-V2-era recipes** — QK-norm (RMSNorm flavor) to stabilize large-batch / high-LR pretraining without warmup babysitting.

## Common mistake

Normalizing along the wrong axis or before the head split. QK-norm must be applied **per head along d_head** (after reshaping), not across the full d_model — normalizing the concatenated projection defeats the purpose and couples heads. Also: don't forget the 1/√d_head scale is still needed (RMSNorm fixes variance, not the √d factor); with cosine/L2 the learnable temperature replaces it.

## See also
- [[entropy-collapse-exploration-control]] — the failure mode QK-norm directly prevents: softmax saturating to one-hot
- [[loss-spikes-training-instability]] — runaway attention logits are a primary spike trigger
- [[rmsnorm]] — the normalizer of choice for the LLM-style QK-norm variant
