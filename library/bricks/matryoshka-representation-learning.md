# Matryoshka Representation Learning (MRL)

**One-liner:** Train one embedding so that every leading prefix (first m dims) is itself a usable representation, letting you truncate at serve time to trade accuracy for storage/latency with no retraining — baked into text-embedding-3, Nomic Embed, Gemini embeddings.

## The formula / definition

Pick a nested set of dimensions M = {8, 16, ..., d} (e.g. {64, 128, 256, ..., 3072}). The total loss is the sum of the task loss applied to each prefix:

L_MRL = Σ_{m∈M} c_m · L( head_m( z[:m] ) , y )

where z ∈ R^d is the full embedding, z[:m] is its first m coordinates, and L is the usual objective (softmax CE for classification, InfoNCE for retrieval). Weights c_m are typically uniform (c_m = 1).

**MRL-E (efficient)** shares one linear head across granularities (weight-tying on the prefix) instead of a separate head per m, removing nearly all the extra training cost. Because losses at coarse m push the most information into the earliest coordinates, the dimensions become **ordered by importance** — like nested matryoshka dolls.

Serving: just slice `z[:m]` and (re)normalize. Optionally pair with adaptive retrieval — shortlist with a short prefix, rerank with the full vector.

## Where it appears

- **Kusupati et al. 2022 (MRL)** — original; matched full-dim ImageNet accuracy using a 16-dim prefix, up to ~14x smaller with negligible loss.
- **OpenAI text-embedding-3** — `dimensions` API param truncates 3072→256/1024; a truncated-and-renormalized 256-dim vector beats the older 1536-dim ada-002.
- **Nomic Embed v1.5 / Gemini embedding / Stella / Jina v3** — ship Matryoshka dims (e.g. 768→512/256/128/64) so users pick the storage/quality point.
- **Vector DBs (ANN)** — coarse prefix for the first-stage ANN scan, full vector for rerank; cuts index size and query cost.

## Common mistake

Forgetting to **re-normalize after truncation**. The full vector is unit-norm, but `z[:m]` is not — for cosine/dot-product retrieval you must renormalize the prefix or your similarities are silently miscalibrated. Related: assuming you can slice *any* dimension count; only the granularities m∈M actually trained on are guaranteed valid (others usually degrade gracefully but aren't optimized).

## See also
- [[embedding-pooling-normalization]] — the L2 renormalization step truncation depends on
- [[ann-vector-search]] — where adaptive prefix-then-full retrieval pays off
- [[infonce-contrastive-loss-with-temperature]] — the per-prefix contrastive objective MRL nests
