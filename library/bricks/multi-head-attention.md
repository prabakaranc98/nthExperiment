# Multi-Head Attention

**One-liner:** Run H attention ops in parallel on linearly-projected d/H-dim subspaces and concatenate, letting heads specialize on different relations; the head is the unit of both interpretability and KV-cache reduction.

## The definition

Given input X ∈ ℝ^{N×d}, split into H heads each of width d_head = d/H. For head i:

Q_i = X W_i^Q,  K_i = X W_i^K,  V_i = X W_i^V   (each ℝ^{d×d_head})

head_i = softmax(Q_i K_iᵀ / √d_head) V_i   (scaled dot-product attention)

MHA(X) = Concat(head_1, …, head_H) W^O,  W^O ∈ ℝ^{d×d}

Total projection FLOPs are the same as one d-wide attention (the W's just block-partition d), but the H separate softmaxes let each head attend to a different mixture of tokens. √d_head (not √d) is the scaling — a frequent off-by-a-factor bug.

## Why heads matter

Each head is a near-independent rank-d_head read/write into the residual stream. Empirically heads specialize: induction heads, positional/previous-token heads, syntactic heads, "name-mover" heads. This makes the head the natural atom for mechanistic interpretability (QK = where-to-look circuit, OV = what-to-copy circuit) and for KV compression (drop/share/quantize per head).

## Where it appears

- **Transformer (Vaswani et al., 2017)** — the original; H=8, d=512, d_head=64
- **GQA / MQA** — keep H query heads but share K/V across groups → shrinks KV cache H/G× with FLOPs flat
- **MLA (DeepSeek-V2/V3)** — replaces per-head K/V with a shared low-rank latent, recovering MHA quality at a fraction of cache
- **Circuits / induction-heads work (Anthropic)** — decomposes attention into per-head QK and OV circuits; the head is the unit of analysis

## Common mistake

Believing more heads = more capacity. With fixed d, adding heads only re-partitions the same parameters into thinner (smaller d_head) subspaces — past a point each head is too low-rank to represent useful relations, and quality drops. Heads buy specialization and parallel relation-mixing, not extra parameters.

## See also
- [[scaled-dot-product-attention]] — the per-head operation MHA runs H of in parallel
- [[gqa]] — shares K/V across query heads; the dominant production variant of MHA
- [[qk-ov-circuits-head-decomposition]] — the interpretability view: each head as a QK + OV circuit
