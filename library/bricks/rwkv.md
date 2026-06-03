# RWKV (Time-Mixing Recurrence)

**One-liner:** An attention-free architecture (RWKV-4→7) that interleaves a token-shift channel-mix with a WKV time-mix behaving like decaying linear attention — trainable in parallel like a transformer, but inferring as a constant-state RNN with O(1) per-token cost and no KV cache.

## The formula / definition

Each block has two sub-layers, both fed a **token shift** (linear interpolation of current and previous token: `μ·x_t + (1-μ)·x_{t-1}`), which gives a cheap 1-token receptive field per layer.

**Time-mix (WKV), RWKV-4 form** — a softmax-weighted sum over the past with exponential decay `w` plus a bonus `u` on the current token:

```
wkv_t = ( Σ_{i<t} e^{-(t-1-i)w + k_i} v_i  +  e^{u + k_t} v_t )
        / ( Σ_{i<t} e^{-(t-1-i)w + k_i}     +  e^{u + k_t} )
out_t = sigmoid(r_t) ⊙ (W_o · wkv_t)
```

R (receptance) gates the output; W,K,V are the named channels. The numerator/denominator are computable as a recurrence with two state vectors (a_t, b_t), hence O(1) memory at inference:

```
a_t = e^{-w} a_{t-1} + e^{k_t} v_t ,   b_t = e^{-w} b_{t-1} + e^{k_t}   (numerically: track a max for stability)
```

**RWKV-5/6 (Eagle/Finch):** scalar decay → **matrix-valued / data-dependent decay** and a vector-state (per-channel) WKV, moving it toward gated linear attention / a linear-attention-with-decay state `S_t = diag(w_t) S_{t-1} + k_t^T v_t`.

**RWKV-7 ("Goose", 2025):** adds a **delta-rule-style** in-context state update — `S_t = S_{t-1}(diag(w_t) - κ_t^T (a_t⊙κ_t)) + v_t^T k_t` — a generalized removal/replace rule that breaks the TC^0 expressivity ceiling of plain linear attention (can track state / do regular-language tasks).

## Where it appears

- **RWKV-4/5/6/7 LLMs** — open multilingual models (incl. World series); the recurrent form enables constant-memory, KV-cache-free streaming inference on edge/CPU.
- **Hybrid stacks & "linear attention" lineage** — sits beside Mamba/GLA/DeltaNet; RWKV-7's delta update mirrors the fast-weight / delta-rule trend in 2024-25 efficient-sequence models.
- **VisualRWKV / RWKV-CLIP / audio** — used as a linear-time backbone where O(N) attention is the bottleneck.

## Common mistake

Thinking the parallel-training form and the RNN-inference form are different models. They are mathematically identical — WKV is a linear recurrence, so it has both a parallel (chunked-scan) form for training throughput and an exact O(1)-state recurrent form for decoding. Also: RWKV ≠ a vanilla RNN — the "R/W/K/V" and exponential decay are what make it competitive, and it does *not* use dot-product softmax attention at all.

## See also
- [[linear-attention]] — WKV is decaying linear attention; same train-parallel / infer-recurrent duality
- [[gated-linear-attention-data-dependent-decay]] — RWKV-5/6's matrix/data-dependent decay is exactly this family
- [[delta-rule-fast-weight-update]] — RWKV-7's in-context state update is a generalized delta rule
