# Diffusion Transformer (DiT)

**One-liner:** Replace the U-Net denoiser with a plain transformer over latent patches, conditioned via adaLN-Zero (timestep + class/text inject scale/shift/gate into LayerNorm), yielding clean compute→FID scaling — the backbone of Sora, SD3, Flux, and Pixart.

## The architecture (Peebles & Xie, ICCV 2023)

1. Encode image with a VAE to a latent z (e.g. 32×32×4 for 256px); patchify into a sequence of tokens (patch size p=2 most common).
2. Add positional embeddings; run N standard transformer blocks (MHA + MLP, no convolutions, no spatial downsampling).
3. Condition every block via **adaLN-Zero**. From the conditioning vector c = embed(t) + embed(y), an MLP regresses six per-channel parameters:

   x = x + α₁ · MHA( γ₁ · LN(x) + β₁ )
   x = x + α₂ · MLP( γ₂ · LN(x) + β₂ )

   "Zero": the gate-producing MLP is initialized so α₁ = α₂ = 0, making each block start as identity — the network begins as the identity function and learns deviations. This is the dominant reason DiT trains stably and scales.
4. Final layer: adaLN modulation → linear → unpatchify → predict noise ε (or v / x₀, per the loss parameterization).

FID drops monotonically with Gflops (more tokens via smaller patch p, or wider/deeper blocks) — the headline scaling result.

## Where it appears

- **Stable Diffusion 3 / Flux** — MM-DiT: separate weights for text vs. image tokens, joint attention across both streams (vs. cross-attention); flow-matching objective.
- **Sora / video DiT** — spacetime patches; DiT over 3D latent tubelets is the published recipe behind it.
- **Pixart-α/Σ** — DiT + cross-attention text conditioning, trained cheaply to T2I quality.

## Common mistake

Thinking adaLN-Zero is just a fancy normalization. The zero-init **gates** (α) are the load-bearing trick: they make residual blocks start as identity, which is what gives DiT its stable, predictable scaling — far more impactful than in-context or cross-attention conditioning, which Peebles & Xie ablate as worse.

## See also
- [[latent-diffusion]] — DiT operates in the VAE latent space LDM defined
- [[vision-transformer-patchification]] — DiT borrows ViT's patchify-then-transformer recipe
- [[flow-matching]] — SD3/Flux pair the DiT backbone with a flow-matching objective
