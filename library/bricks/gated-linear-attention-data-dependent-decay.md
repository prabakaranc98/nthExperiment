# Gated Linear Attention & Data-Dependent Decay

**One-liner:** Replace softmax attention's exact recall with a recurrent matrix-valued memory state S that is multiplicatively decayed by a (often input-dependent) forget gate each step — the single knob (`S_t = G_t ⊙ S_{t-1} + k_t v_t^T`) along which RetNet's fixed scalar γ, HGRN's scalar gate, GLA's per-channel gate, and Mamba's Δ-controlled decay all differ.

## The formula / definition

Linear attention writes a state and reads it back without softmax:

    S_t = S_{t-1} + k_t v_t^T          (state ∈ R^{d_k × d_v})
    o_t = q_t^T S_t / (q_t^T z_t)       (z_t = running sum of keys, optional normalizer)

This is a pure RNN over a matrix memory — but with no forgetting, old keys never decay and the state saturates. **Decay/gating fixes this by inserting a multiplicative term:**

    S_t = Diag(α_t) S_{t-1} + k_t v_t^T

The decay α_t (in (0,1)) is the discriminating axis:

- **RetNet:** α = γ, a *fixed scalar* per head (data-independent). Closed-form, very fast.
- **HGRN / Gated RetNet:** α_t a *scalar* but *data-dependent* (function of x_t).
- **GLA (Yang et al. 2024):** α_t a *per-channel vector* gate, α_t = σ(x_t W_α) → richer selectivity, requires the chunkwise kernel to stay efficient.
- **Mamba (S6):** α_t = exp(−Δ_t A) with Δ_t = softplus(x_t W_Δ) data-dependent → selective SSM; the same forget mechanism in continuous-time/SSM clothing.
- **Mamba-2 / DeltaNet:** decay tied to a structured (scalar-times-identity or generalized) transition, exposing the **state-space duality** with attention.

Gating is what makes these models *selective*: a near-1 gate keeps memory, a near-0 gate flushes it conditioned on content.

## Where it appears

- **GLA** — 2D per-channel data-dependent gate + a hardware-aware chunkwise-parallel form so training is matmul-bound, not sequential.
- **Mamba / Mamba-2** — Δ_t is exactly the data-dependent decay; "selective" = the gate depends on input, unlike LTI SSMs (S4).
- **RetNet** — fixed multi-scale γ per head gives a parallel + recurrent + chunkwise triple form with O(1) inference state.
- **HGRN1/2, RWKV-6/7** — gated linear recurrences; RWKV-7 adds a delta-rule-style state update on top of decay.
- **Hybrid stacks (Jamba, Zamba, Falcon-H1, Hymba, 2024–25)** — interleave a few full-attention layers with many gated-linear/SSM layers for cheap long context.

## Common mistake

Conflating the **gate (decay, multiplicative, controls forgetting)** with the **input/output gates of an LSTM** or with the **write key k_t**. Decay scales the *existing* state toward zero; it does not select what to write. Also: assuming data-dependent decay breaks parallel training — it doesn't, the chunkwise form keeps it matmul-heavy; you only lose the simplest closed-form scan that fixed-γ RetNet enjoys.

## See also
- [[linear-attention]] — the un-gated base this augments
- [[selective-state-space-models-mamba]] — Δ_t is the same decay gate in SSM form
- [[state-space-duality]] — why decayed linear attention and structured SSMs are the same object
