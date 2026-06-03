# Hadamard / Rotation-Based Quantization (QuaRot, SpinQuant)

**One-liner:** Multiply weights and activations by orthogonal (often Hadamard) matrices that are computationally inert (Q Qᵀ = I) but spread energy across channels, killing the massive-magnitude outlier features that otherwise dominate the quantization range — now the standard front-end for 4-bit weight+activation (W4A4) pipelines.

## The key insight

A linear layer y = Wx is invariant under any orthogonal Q: y = (W Q)(Qᵀ x). Pick Q to make both W Q and Qᵀ x outlier-free. A random/Hadamard rotation mixes channels so no single dimension carries a huge spike — variance is spread, max/mean magnitude ratio drops, so a per-tensor or per-token quantizer wastes far fewer levels on rare extremes. After rotation the tensors are near-Gaussian and quantize cleanly at 4 bits.

Hadamard matrices H (entries ±1/√d, H Hᵀ = I) are the practical choice: the Walsh-Hadamard transform applies Q in O(d log d) on the fly with no stored matrix, fusing into the kernel essentially for free.

Two ways to absorb the rotation:
- **Fused/offline:** fold Q into adjacent weight matrices (e.g. into Wₖ, Wᵥ, down-proj input, RMSNorm scale) so it costs nothing at inference. Used around residual stream and MLP.
- **Online:** apply a Hadamard transform inside the kernel just before quantizing an activation that has no foldable partner (e.g. before the down-projection, or to keys/values for KV-cache quant).

## Where it appears

- **QuaRot (2024)** — rotates the entire transformer (residual stream, attention, MLP) with Hadamard matrices so *all* of weights, activations, and KV-cache quantize to 4 bits; outlier-free end-to-end, no mixed-precision FP16 escape hatch for outlier channels.
- **SpinQuant (2024)** — learns the rotation matrices (optimized on Stiefel manifold via Cayley SGD) instead of fixing them to random Hadamard, closing most of the remaining gap to FP16 at W4A4.
- **QuIP / QuIP# (2-bit)** — incoherence processing via random orthogonal / Hadamard transforms is the same trick pushed to extreme sub-4-bit weight-only quant.
- KV-cache quantization and FP4/MXFP4 inference kernels increasingly bundle an online Hadamard step before casting.

## Common mistake

Thinking the rotation removes information or is an approximation — it is exactly invariant (Q Qᵀ = I), the output is mathematically unchanged in full precision. The benefit is purely statistical: it reshapes the value distribution so the *quantizer* incurs less error. A second mistake: not realizing rotations must be placed at the right computational invariances (around RMSNorm, across the residual stream, between fusable matmuls) — a rotation you can't fold or cheaply apply online buys you nothing.

## See also
- [[activation-outliers-smoothquant]] — the older fix (per-channel scaling migration) for the same outlier problem; rotation supersedes it at 4-bit
- [[quantization]] — the base operation rotation is a pre-conditioner for
- [[extreme-sub-4-bit-quantization]] — incoherence/Hadamard processing is what makes 2-bit (QuIP#) viable
