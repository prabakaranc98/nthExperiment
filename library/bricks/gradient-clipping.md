# Gradient Clipping

**One-liner:** Rescale the gradient — almost always by its global L2 norm — down to a threshold c so no single update can blow up, capping the damage from rare large gradients; the default guardrail in every large-scale run.

## The formula / definition

Compute the global norm over *all* parameters concatenated, then scale the whole vector if it exceeds the threshold c:

```
‖g‖₂ = sqrt( Σ_p ‖g_p‖₂² )          # one norm across all params
g ← g · min(1, c / ‖g‖₂)            # only shrinks; direction preserved
```

Because the scale factor is shared, the update **direction is unchanged** — clipping only ever reduces magnitude (when ‖g‖₂ > c) and is a no-op otherwise. Typical c ∈ [0.5, 1.0] for transformer pretraining.

Variants:
- **Value clipping:** g ← clamp(g, −v, +v) per coordinate. Distorts direction; rarely used for LLMs.
- **Per-parameter / per-layer norm clipping:** clips each tensor's norm independently — also distorts the global direction; generally avoided.
- **Adaptive Gradient Clipping (AGC, NFNets):** clip by the ratio ‖g_p‖/‖θ_p‖, enabling normalizer-free training.
- **AutoClip / percentile clipping:** set c from a running percentile of recent observed ‖g‖₂ instead of a fixed constant.

## Where it appears

- **GPT-2 / GPT-3 / most LLM pretraining** — global-norm clip at 1.0 as standard recipe, paired with warmup and bf16 master weights.
- **Loss-spike mitigation** — a large gradient from a rare/bad batch is the classic spike trigger; clipping bounds its single-step impact (often combined with batch-skipping on recovery).
- **RNN/LSTM era (Pascanu et al. 2013)** — the original motivation: exploding gradients through long BPTT chains.
- **DP-SGD (differential privacy)** — *per-example* clipping bounds each sample's sensitivity before adding noise — here clipping is the privacy mechanism, not just stability.
- **NFNets (Brock et al. 2021)** — AGC replaces BatchNorm's implicit stabilization.

## Common mistake

Thinking clipping changes the optimization direction or is a substitute for fixing the underlying instability. Global-norm clipping preserves direction (it is just a scalar shrink), but a frequently-active clip is a *symptom*: lower LR, add warmup, or bound the blowing-up quantity (QK-norm for attention logits, z-loss for output logits). Also: clipping per-tensor or per-coordinate instead of by global norm silently rotates the update.

## See also
- [[training-stability]] — clipping is one item in the warmup / QK-norm / z-loss stability toolkit
- [[loss-spikes-training-instability]] — a rare huge-gradient batch is the spike clipping is meant to contain
- [[differential-privacy]] — DP-SGD repurposes per-example clipping to bound sensitivity
