# Muon Optimizer

**One-liner:** Momentum SGD for 2D weight matrices whose update is orthogonalized via a few Newton-Schulz iterations (cheap approximate `U V^T` from the gradient's SVD), matching or beating AdamW at lower wall-clock and memory on hidden-layer params.

## The formula / definition

For a 2D weight `W`, accumulate momentum on the gradient, then *orthogonalize* the momentum buffer before stepping:

```
M_t = mu * M_{t-1} + G_t                 # heavy-ball momentum (mu ~ 0.95)
O_t = NewtonSchulz5(M_t)                  # ~5 iters -> O_t ≈ U V^T  (semi-orthogonal)
W_t = W_{t-1} - eta * O_t
```

Newton-Schulz quintic iteration on normalized `X = M / ||M||_F` pushes all singular values toward 1:
`X <- a*X + b*(X X^T)X + c*(X X^T)^2 X`, coefficients `(a,b,c) ≈ (3.4445, -4.7750, 2.0315)` tuned so the polynomial maps σ∈(0,1] near 1; runs in bf16, no exact SVD. Effectively a per-layer *spectral/whitening* step: replaces the update with its closest orthogonal matrix `UV^T`.
Scale `eta` by `sqrt(max(fan_in,fan_out))` (or `0.2*sqrt(d)`) so the RMS update matches Adam — this is what makes the LR transferable across shapes.

## Where it appears

- **Keller Jordan's NanoGPT speedrun (2024)** — origin; set CIFAR/GPT speed records vs AdamW.
- **Moonshot Kimi K2 (2025)** — Muon scaled to a ~1T-param MoE via *MuonClip* (adds QK-clipping to tame logit/attention blowups at scale), the first frontier-scale validation.
- **Microsoft Dion, Essential AI, various 2025 scaling studies** — distributed/sharded Muon variants and scaling-law fits.

## Common mistake

Applying Muon to *everything*. It is only for 2D hidden weight matrices. Embeddings, the unembedding/LM head, biases, and all norm gains/scalars are kept on AdamW — they're 1D or have row-meaning that orthogonalization destroys. Muon is a hybrid optimizer, not a drop-in global replacement.

## See also
- [[muon-spectral-norm-newton-schulz]] — the orthogonalization primitive and spectral-norm view in detail
- [[shampoo-soap]] — the preconditioner Muon approximates cheaply (Muon ≈ momentum-Shampoo without accumulator inversions)
- [[adamw]] — the baseline it competes with and still uses for 1D params
