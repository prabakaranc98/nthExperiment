# Residual / Skip Connections

**One-liner:** y = x + F(x); an identity shortcut around a block so the layer learns a residual rather than a full transform — this makes 100+ layer nets trainable and turns the per-token activation into a persistent additive "residual stream."

## The formula / definition

A block computes a residual function F and adds it back to its input:

  y = x + F(x)

so F only needs to learn the *correction* to the identity, and ∂y/∂x = I + ∂F/∂x — the identity term keeps the gradient from vanishing through depth (no product of many <1 Jacobians). Pre-norm transformers place norm *inside* the branch so the skip path stays clean:

  x_{l+1} = x_l + Attn(LN(x_l))
  x_{l+2} = x_{l+1} + MLP(LN(x_{l+1}))

This makes the hidden state a running sum: x_L = x_0 + Σ_l F_l(...). Every block reads from and writes to this shared "residual stream" — the linear backbone interpretability work decomposes models around.

## Where it appears

- **ResNet (He et al., 2015)** — origin; identity shortcut let 152-layer CNNs beat shallow nets, solving the degradation problem (deeper got *worse*, not just overfit)
- **Transformers** — every attention/MLP sublayer is residual; pre-norm + residual is what makes deep LLMs stable to train
- **Diffusion U-Nets / DiT** — residual blocks plus long skip connections across the encoder/decoder
- **LoRA** — the adapter ΔW = BA is itself a residual added to frozen W: y = Wx + BAx

## Common mistake

Confusing the residual *connection* (the y = x + F(x) mechanism) with the residual *stream* (the running-sum activation tensor it creates). Also: thinking the skip "skips computation" — it adds the input back; F is always computed. And in pre-norm transformers the residual path must stay un-normalized, or you lose the clean identity that enables the logit/feature-additive view.

## See also
- [[residual-stream]] — the additive backbone these connections create, read/written by every block
- [[pre-norm-vs-post-norm-vs-sandwich-norm]] — where the norm sits relative to the skip determines stability
- [[initialization-residual-depth-scaling]] — scaling F at init (e.g. 1/√L) to keep the residual variance controlled with depth
