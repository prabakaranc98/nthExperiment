# Hybrid Attention-SSM Architectures

**One-liner:** Interleave a few full-attention layers among many SSM / linear-attention layers (Jamba, Zamba, Samba, Nemotron-H) so the rare attention layers restore exact recall, copying, and in-context retrieval while the model stays mostly sub-quadratic with a constant-size recurrent state.

## The key insight

A pure SSM (Mamba) compresses history into a fixed-size state `h_t ∈ R^d`, so it cannot losslessly recall arbitrary past tokens — associative recall is bounded by state size, not sequence length. Full softmax attention has an unbounded KV cache and recalls exactly but costs O(N²) compute and O(N) memory per layer.

Hybrids exploit that you don't need exact recall in *every* layer. Use a layout like:

    [SSM, SSM, SSM, ..., ATTN, SSM, SSM, ..., ATTN, ...]
    ratio ≈ 1 attention layer per 6–8 SSM/GLA layers

The SSM layers do cheap sequence mixing in constant memory; the sparse attention layers (often the only thing holding a KV cache) act as a global lookup that recovers what the recurrent state dropped. Net: near-linear compute, much smaller KV cache (only attn layers cache), recall close to a full transformer.

## Where it appears

- **Jamba (AI21, 2024)** — Mamba-2 + Transformer + MoE blocks, ~1:7 attn:Mamba ratio; 256K context, small KV cache vs dense.
- **Zamba / Zamba2 (Zyphra)** — Mamba backbone with a *single shared* attention block reused across depth (weight-tied) to cut params.
- **Samba (Microsoft, 2024)** — Mamba + sliding-window attention + MLP, extrapolates far beyond train length.
- **Nemotron-H / Hymba (NVIDIA, 2024–25)** — mostly Mamba-2 with a few full-attention layers; Hymba runs attention and SSM heads *in parallel* within a layer.
- **MiniMax-01 (2025)** — lightning (linear) attention with periodic softmax-attention layers at scale.

## Common mistake

Believing the SSM layers handle long-context recall and attention is just a "bonus." It's the reverse: ablating the handful of full-attention layers collapses retrieval/copying (NIAH) almost to chance — the attention layers carry the exact-recall load, and placement/count matters more than total layer budget.

## See also
- [[selective-state-space-models-mamba]] — the sub-quadratic backbone these hybrids are built on
- [[associative-recall-the-recall-state-size]] — why fixed-state SSMs need attention layers to recall
- [[sliding-window-local-attention]] — the cheap attention variant often paired with SSM layers
