# VQ-VAE / Discrete Visual Tokenizers

**One-liner:** Encode an image/video into a grid of discrete codebook indices via nearest-neighbor vector quantization, enabling autoregressive or masked-token modeling over the tokens; codebook collapse and FSQ/lookup-free quantization recur.

## The formula / definition

Encoder z_e(x) ∈ ℝ^{H×W×D}. Quantize each spatial vector to its nearest codebook entry e_k:

  z_q = e_k,  k = argmin_j ‖z_e(x) − e_j‖₂

Decoder reconstructs x̂ = decode(z_q). argmin is non-differentiable, so gradients use the **straight-through estimator** (copy ∇ from z_q to z_e). Loss:

  L = ‖x − x̂‖ + ‖sg[z_e] − e‖₂² + β‖z_e − sg[e]‖₂²

where sg = stop-gradient. Term 2 = codebook loss (move codes to encoder outputs; often replaced by EMA updates). Term 3 = commitment loss (β≈0.25, keep encoder near its chosen code). VQ-GAN adds a perceptual (LPIPS) + adversarial (PatchGAN) loss for sharp reconstructions.

**FSQ (Finite Scalar Quantization, 2023):** drop the codebook entirely. Project to d dims (d≈5), round each to one of L levels → implicit codebook of L^d codes. No collapse, no aux losses, no STE bookkeeping.

## Where it appears

- **VQ-GAN / Taming Transformers** — perceptual+adversarial codec feeding a GPT-style AR prior; the template for token-based image generation
- **MaskGIT / MUSE / MAGVIT-v2** — masked parallel decoding over visual tokens; MAGVIT-v2 uses **lookup-free quantization (LFQ)** (binary per-dim codes) to scale the codebook to ~2^18 and beat diffusion on ImageNet
- **DALL·E 1, Parti, Chameleon, Emu3** — discrete image (and video) tokens in a shared vocabulary so one transformer models text+image autoregressively
- **Neural audio codecs (SoundStream, EnCodec, DAC)** — RVQ (residual VQ) stacks of the same quantizer for audio LMs

## Common mistake

Thinking a bigger codebook = better. Naively scaling the codebook causes **codebook collapse**: most entries are never selected (dead codes), so effective vocabulary stays tiny and reconstruction stalls. The fixes that work are structural — EMA codebook updates, dead-code reinit/restart, ℓ2-normalized low-dim codes (ViT-VQGAN), or sidestepping the codebook with FSQ/LFQ — not just raising K.

## See also
- [[gumbel-softmax-straight-through-estimator]] — the discretization gradient trick that makes VQ trainable
- [[vae-the-elbo-reparameterization-trick]] — the continuous-latent ancestor VQ replaces with a discrete prior
- [[autoregressive-visual-generation]] — the dominant downstream use of these token grids
