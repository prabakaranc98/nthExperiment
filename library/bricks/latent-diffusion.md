# Latent Diffusion (LDM)

**One-liner:** Run the diffusion process in the compressed latent space z = E(x) of a frozen, pretrained autoencoder instead of in pixel space — cutting spatial dimensions ~8x per side (~48x total) and thus compute, while a perceptual+adversarial decoder D restores fidelity; the architecture behind Stable Diffusion and most production image/video generators.

## The formula / definition

**Stage 1 — perceptual autoencoder (trained first, then frozen):** encode z = E(x), decode x̂ = D(z). For 512x512 RGB at downsample factor f=8, z is 64x64xC. Trained with an L1/L2 + LPIPS perceptual loss + a patch-GAN adversarial term, plus a *mild* regularizer on the latent: either a low-weight KL toward N(0,I) (KL-reg, continuous, the SD choice) or a vector-quantization codebook (VQ-reg). The KL weight is tiny — this is an autoencoder with a slight bottleneck, **not** a high-fidelity VAE prior.

**Stage 2 — diffusion in latent space:** train an ordinary DDPM/score model on z₀ = E(x), not on x:
  L_LDM = E_{t, x, ε} ‖ ε − ε_θ(z_t, t, c) ‖²,  z_t = √ᾱ_t z₀ + √(1-ᾱ_t) ε
Conditioning c (text, class, image) enters the denoiser via cross-attention. Sample z_T ~ N(0,I), run the reverse process (DDIM/EDM sampler) to ẑ₀, then **decode once**: x̂ = D(ẑ₀). Pixels are touched only by E (once, at training) and D (once, at sampling).

## Where it appears

- **Latent Diffusion Models (Rombach et al. 2022) / Stable Diffusion 1.x–2.x** — the original: KL-reg f=8 autoencoder + UNet denoiser + cross-attention text conditioning (CLIP/OpenCLIP).
- **SDXL** — larger UNet, two text encoders, refiner; same latent recipe.
- **Stable Diffusion 3 / FLUX** — replace the UNet with a [[diffusion-transformer]] and DDPM with [[flow-matching]] (rectified flow), still operating on VAE latents.
- **Video (Sora-style, Movie Gen, CogVideoX)** — a 3D spatiotemporal latent (a video autoencoder compresses time too); diffusion runs on the compressed video latent.
- **Audio (Stable Audio)** — diffusion over a learned audio latent.

## Common mistake

Conflating the LDM autoencoder with a generative VAE. It is **deliberately under-regularized** — the KL weight is so small that the latent is nearly an unconstrained autoencoder code, not a sampleable N(0,I) prior. You cannot sample z ~ N(0,I) and decode to get an image; the *diffusion model* learns the latent distribution. The autoencoder's only job is faithful reconstruction at high compression. Second mistake: forgetting the latent is scaled by a fixed factor (SD's ~0.18215) so its variance ≈ 1 before diffusion training.

## See also
- [[ddpm]] — LDM is DDPM run on z instead of x; identical noise-prediction objective
- [[vae-the-elbo-reparameterization-trick]] — the encoder/decoder, but with a near-zero KL weight (perceptual AE, not a true VAE)
- [[diffusion-transformer]] — the modern denoiser backbone that replaced the UNet inside the latent
