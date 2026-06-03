# Hard Negative Mining

**One-liner:** Train contrastive retrievers against negatives that are semantically close to the query (high similarity, wrong answer) rather than random in-batch ones — this sharpens the decision boundary and is the single biggest lever on retriever quality, but must be guarded against false negatives.

## The key insight

InfoNCE loss is dominated by the hardest (highest-scoring) negatives:

L = −log [ exp(s(q,d⁺)/τ) / (exp(s(q,d⁺)/τ) + Σ_{d⁻} exp(s(q,d⁻)/τ)) ]

Random negatives have s(q,d⁻) ≈ 0, contribute ~nothing to the gradient. **Hard** negatives have s(q,d⁻) near s(q,d⁺), so they produce the largest gradient signal and force the model to learn fine-grained distinctions. Gain per negative >> gain from simply adding more easy negatives.

**Standard recipe (hardest part is mining the negatives, not the loss):**
1. Train an initial retriever (or use BM25).
2. For each query, retrieve top-k; take the highest-ranked **non-gold** passages as hard negatives.
3. Retrain. Optionally iterate (mine with the new model → "ANCE" style refresh).

## Where it appears

- **DPR / ANCE / RocketQA** — DPR mixed BM25 hard negatives + in-batch; ANCE re-mines hard negatives from an async-refreshed ANN index every checkpoint; RocketQA added cross-encoder denoising to filter false negatives.
- **E5 / BGE / GTE / NV-Embed (2023–2025 embedding models)** — multi-stage: weak contrastive pretrain, then SFT on labeled pairs with mined hard negatives, often with an LLM/cross-encoder to score and discard likely false negatives.
- **ColBERT & late-interaction training** — hard negatives critical because token-level scoring is easy to saturate with random negatives.
- **RLHF reward models / preference data** — "hard" rejected responses (close to chosen) analogously drive the most useful gradient.

## Common mistake

**False negatives.** Top-k mined "negatives" are often actually relevant (just unlabeled) — penalizing them teaches the model the wrong boundary and *degrades* recall. Mitigate: score candidates with a stronger cross-encoder/LLM and drop the near-gold ones, skip the rank-1 hits, or use a margin/score threshold. Mining negatives that are *too* hard without denoising is worse than random negatives.

## See also
- [[infonce-contrastive-loss-with-temperature]] — the loss whose gradient hard negatives dominate
- [[dense-retrieval]] — the system whose quality this most directly controls
- [[cross-encoder-reranking]] — the denoiser/teacher used to filter false negatives
