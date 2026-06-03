# Scaled Dot-Product Attention

**One-liner:** Softmax(QKᵀ/√dₖ)V — a value-weighted average where weights come from query-key dot products, with the √dₖ scaling holding logits in softmax's non-saturating regime; O(n²d) FLOPs and an O(n²) memory wall.

## The formula / definition

For queries Q ∈ ℝ^{n×dₖ}, keys K ∈ ℝ^{m×dₖ}, values V ∈ ℝ^{m×dᵥ}:

  Attention(Q,K,V) = softmax(QKᵀ / √dₖ + M) V

- QKᵀ ∈ ℝ^{n×m} = pairwise similarity logits (one query row attends over m keys)
- softmax is taken **row-wise** → each query gets a probability distribution over keys
- M = mask (−∞ for causal/padding positions, added pre-softmax)
- Output ∈ ℝ^{n×dᵥ}: row i is Σⱼ pᵢⱼ vⱼ, a convex combination of value vectors

**Why √dₖ:** if q,k entries are ~unit-variance and independent, qᵀk has variance dₖ. Dividing by √dₖ renormalizes logit variance to ~1, preventing the softmax from saturating into a near-one-hot argmax with vanishing gradients as dₖ grows.

**Cost:** QKᵀ and PV are each O(n·m·d); for self-attention (n=m) that's O(n²d) compute and the score matrix is O(n²) — the quadratic memory/IO bottleneck that everything downstream attacks.

## Where it appears

- **Transformer (Vaswani et al., 2017)** — the core operation; multi-head attention runs h of these in parallel on projected subspaces
- **FlashAttention / FlashAttention-2/3** — computes the *exact* same SDPA but never materializes the n×n matrix, using the online-softmax recurrence + tiling
- **Every LLM inference stack** — the KV cache stores K,V so each decode step is one query against all past keys
- **`torch.nn.functional.scaled_dot_product_attention`** — the fused PyTorch op that dispatches to Flash/mem-efficient kernels

## Common mistake

Confusing the scaling factor with the head dimension globally: it is √dₖ (the **per-head** key dimension d_model/h), not √d_model. Also: thinking the scale is cosmetic — without it, deep/wide models train unstably because softmax saturates and gradients through it die.

## See also
- [[softmax]] — the normalization whose saturation the √dₖ scaling prevents
- [[flash-attention]] — IO-aware exact computation of this same operation
- [[multi-head-attention]] — runs many SDPA heads in parallel subspaces
