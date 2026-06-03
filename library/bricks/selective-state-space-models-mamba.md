# Selective State-Space Models / Mamba

**One-liner:** SSMs whose (B, C, Δ) are made functions of the input — content-based gating lets the recurrence selectively remember or forget tokens, closing the quality gap with attention at O(N) cost and O(1) inference state.

## The formula / definition

A standard SSM is a linear recurrence with continuous params (A, B), discretized by step Δ:

  Ā = exp(ΔA),  B̄ = (ΔA)^{-1}(exp(ΔA) − I)·ΔB  (zero-order hold)
  h_t = Ā h_{t−1} + B̄ x_t,   y_t = C h_t   (+ D x_t skip)

**LTI SSM (S4):** A, B, C, Δ are fixed across time → the recurrence is a long convolution, trainable via FFT. But being time-invariant, it cannot condition on content.

**Selection (Mamba):** make B, C, Δ functions of the input x_t:
  Δ_t = softplus(Linear(x_t)),  B_t = Linear(x_t),  C_t = Linear(x_t)
  h_t = Ā_t h_{t−1} + B̄_t x_t   with  Ā_t = exp(Δ_t A)

A stays a learned per-channel scalar/diagonal. Large Δ_t → reset/absorb current token; small Δ_t → ignore it and carry state (a soft gate, analogous to GRU/LSTM gates). This breaks time-invariance, so no FFT convolution — instead use a **hardware-aware parallel scan** in SRAM (Mamba-1) or chunked matmul form (Mamba-2 / SSD), keeping it O(N) and GPU-efficient.

State is a fixed-size matrix (size N_state × d), independent of sequence length → recall capacity is bounded by state size, not free like an attention KV cache.

## Where it appears

- **Mamba (Gu & Dao, 2023)** — selective scan + gated MLP block; matched Transformers at <3B on language, dominant on DNA/audio.
- **Mamba-2 / State-Space Duality (2024)** — restricts A to scalar-times-identity, recasts selective SSM as a form of masked linear attention computed in chunked/matmul form; much faster, larger state.
- **Hybrid stacks (Jamba, Zamba, NVIDIA Nemotron-H, Falcon-Mamba, IBM Bamba, 2024-2026)** — interleave a few full-attention layers with many Mamba layers for long-context throughput while preserving exact recall.

## Common mistake

Thinking selectivity just means "input-dependent A." The decisive, efficiently-implementable choice is making **Δ, B, C input-dependent** (A stays structured/diagonal). Also: assuming Mamba matches attention on hard associative recall — its fixed state caps recall, which is exactly why frontier systems go hybrid.

## See also
- [[linear-time-invariant-ssm]] — the non-selective S4 predecessor selection generalizes
- [[state-space-duality]] — Mamba-2's bridge to linear attention and chunked matmul form
- [[hardware-aware-parallel-scan]] — the scan kernel that makes selective SSMs fast on GPUs
