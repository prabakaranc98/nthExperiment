# Data-Constrained Scaling & Repetition Laws

**One-liner:** Extension of Chinchilla to repeated data — each extra epoch on the same tokens has exponentially decaying value, with up to ~4 epochs nearly as good as fresh tokens and ~16+ epochs adding almost nothing; the binding constraint once unique web text runs out.

## The formula (Muennighoff et al., NeurIPS 2023)

Replace raw counts N, D in Chinchilla with *effective* counts that discount repetition. For D_C unique tokens repeated over D total tokens (R_D = D/D_C − 1 extra epochs):

D' = D_C + U_D · R_D* · (1 − e^{−R_D / R_D*})

The repeated tokens contribute with exponentially decaying marginal value, governed by decay constant R_D* (fit ≈ 15 for tokens). An analogous form discounts *excess parameters* N. Loss is then the Chinchilla form L(N', D') evaluated on the effective counts.

**Rule of thumb:** ~4 epochs ≈ as good as that many fresh unique tokens (value loss < ~few %); past ~16 epochs repeating is nearly worthless and you should spend compute elsewhere (more params, more filtering, or stop).

## Where it appears

- Muennighoff et al. "Scaling Data-Constrained Language Models" — the foundational fit; also shows code-mixing and filtering as alternatives to repetition.
- Llama 3 / DeepSeek / Qwen pretraining — over-training far past Chinchilla-optimal forces multi-epoch reasoning on high-quality subsets; these laws bound how much to repeat math/code.
- Synthetic data & rephrasing pipelines — rephrasing (e.g. WRAP, "tinystories"-style) is partly a way to manufacture *fresh* effective tokens rather than re-epoching stale ones.

## Common mistake

Assuming repeated tokens are worthless (1 epoch only) — they are not; up to ~4 epochs is nearly free. The opposite mistake is treating epochs as fully equivalent to fresh data: marginal value decays exponentially, and excessive repetition eventually drives memorization without generalization gains.

## See also
- [[scaling-laws]] — the Chinchilla baseline this generalizes when data is unlimited
- [[synthetic-data-web-rephrasing]] — generating effective fresh tokens as the alternative to repetition
- [[memorization-vs-generalization]] — what over-repeating data trades into
