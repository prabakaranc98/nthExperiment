# InfoNCE / Contrastive Loss with Temperature

**One-liner:** Cross-entropy over similarity logits that pulls one positive together against many negatives, scaled by a temperature τ that controls how sharply the loss concentrates penalty on the hardest negatives.

## The formula / definition

For an anchor with embedding query q, one positive k+, and a set of negatives {k-}:

L = -log [ exp(sim(q, k+)/τ) / Σ_i exp(sim(q, k_i)/τ) ]

where sim is usually cosine similarity (L2-normalized dot product) and the sum runs over the positive plus all negatives. This is exactly softmax cross-entropy where the positive is the correct "class." In SimCLR/CLIP, the batch supplies the negatives (in-batch negatives), and the loss is computed symmetrically (image->text and text->image, averaged).

τ → 0: gradient dominated by the single hardest negative (sharp, high-penalty, can be unstable). τ → ∞: uniform, weak signal. Typical: 0.05–0.2 (SimCLR ~0.1, CLIP learns it, clamped). InfoNCE is a lower bound on mutual information I(q; k+); maximizing it ≈ maximizing MI between views.

## Where it appears

- CLIP — symmetric image-text InfoNCE over a batch; temperature is a learned, exp-parameterized scalar clamped at 100.
- SimCLR / MoCo — self-supervised vision; MoCo decouples negatives into a momentum-updated queue (memory bank) so batch size need not equal negative count.
- Dense retrieval (DPR, E5, GTE) — query-passage contrastive training with in-batch + mined hard negatives.

## Common mistake

Forgetting that τ rescales the gradient, not just the logits — it is not a free knob. Lower τ implicitly up-weights hard negatives and demands harder negatives / larger batches; too low causes collapse or instability. Also: omitting L2 normalization so sim becomes unbounded dot product, which makes τ meaningless.

## See also
- [[clip-contrastive-vision-language-pretraining]] — the canonical large-scale symmetric InfoNCE
- [[siglip-sigmoid-contrastive-loss]] — replaces the softmax/normalizer with per-pair sigmoid, removing the batch-wide partition function
- [[hard-negative-mining]] — what controls the effective difficulty the temperature trades off against
