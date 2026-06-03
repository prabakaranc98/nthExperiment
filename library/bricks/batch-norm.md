# Batch Normalization

**One-liner:** Normalize each activation channel across the batch dimension using mini-batch mean/variance during training and a running EMA at inference; the historical anchor of the normalization literature, still default in CNNs/vision but largely displaced by LayerNorm/RMSNorm in transformers.

## The formula / definition

For a channel/feature, over the mini-batch B (and spatial dims in conv nets):
- μ_B = mean(x),  σ²_B = var(x)
- x̂ = (x − μ_B) / √(σ²_B + ε)
- y = γ · x̂ + β   (learnable scale γ, shift β)

Train: use batch stats μ_B, σ²_B. **Inference:** use running estimates μ_run, σ²_run = EMA over training batches (momentum, e.g. 0.1). This train/eval discrepancy is the crux of BN.

## Why it works (contested)

Original claim (Ioffe & Szegedy, 2015): reduces "internal covariate shift." Santurkar et al. (2018) refuted this — the real effect is **smoothing the loss landscape** (smaller Lipschitz/β-smoothness), enabling higher learning rates. BN also injects stochastic regularization via batch-dependent noise.

## Where it appears

- ResNet / VGG / Inception — BN after conv, before ReLU; enabled training of very deep CNNs.
- EfficientNet, ConvNeXt-era vision — still standard (ConvNeXt swaps to LayerNorm).
- GANs — BN destabilizes; replaced by InstanceNorm / spectral norm.
- Transformers / LLMs — BN is *not* used; LayerNorm/RMSNorm dominate (no batch dependence, works at batch size 1, sequence-friendly).

## Common mistake

Forgetting that BN behaves differently in train vs eval mode (`model.eval()` / `training=False`), and that it breaks with **small or correlated batches**. Tiny batch sizes give noisy stats → use GroupNorm/LayerNorm instead. Also: BN couples examples within a batch, so it leaks information across samples and is unsafe under sequence/causal or per-example settings.

## See also
- [[layer-norm]] — normalizes per-example over features; no train/eval gap, the transformer default
- [[rmsnorm]] — cheaper LayerNorm variant used in modern LLMs
- [[training-stability]] — BN's real benefit is landscape smoothing, enabling larger learning rates
