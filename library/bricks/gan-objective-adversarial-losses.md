# GAN Objective & Adversarial Losses

**One-liner:** A generator-discriminator minimax game (non-saturating / hinge losses, R1 gradient penalty, spectral norm) that learns to draw from p_data without an explicit likelihood, yielding sharp samples — still load-bearing as a perceptual loss in VQGAN tokenizers and one-step diffusion distillation.

## The formula

Original minimax objective (Goodfellow et al., 2014):

min_G max_D  E_{x∼p_data}[log D(x)] + E_{z∼p_z}[log(1 − D(G(z)))]

At the optimal D, this minimizes the Jensen-Shannon divergence between p_data and p_G.

**Non-saturating generator loss** (the actual default): instead of minimizing log(1−D(G(z))) (which has vanishing gradients when D is confident), maximize log D(G(z)). Same fixed point, usable gradients early in training.

**Hinge loss** (BigGAN, SAGAN, VQGAN — the modern default):
- L_D = E[max(0, 1 − D(x))] + E[max(0, 1 + D(G(z)))]
- L_G = −E[D(G(z))]

## Stabilizers (these matter more than the loss choice)

- **R1 gradient penalty** (Mescheder et al., 2018): add (γ/2)·E[‖∇_x D(x)‖²] on real samples. The single most reliable stabilizer; used in StyleGAN2/3.
- **Spectral normalization** (Miyato et al., 2018): divide each weight matrix by its largest singular value to enforce a 1-Lipschitz D — controls the gradient the discriminator can pass back.
- **Wasserstein / WGAN-GP**: replace JS with the Wasserstein-1 distance estimated via a 1-Lipschitz critic; smoother loss landscape, less mode collapse.

## Where it appears

- **VQGAN / VQ-VAE-2 decoders** — a patch discriminator + adversarial loss is added on top of perceptual (LPIPS) + L1 to make discrete-token image reconstructions sharp; this is why latent-diffusion VAEs and image tokenizers look crisp rather than blurry.
- **Diffusion / consistency distillation** — adversarial losses (ADD, Adversarial Diffusion Distillation in SDXL-Turbo, DMD2, Hyper-SD) sharpen 1–4 step student samples that an MSE distillation loss alone leaves smeared.
- **StyleGAN2/3, BigGAN** — the canonical high-fidelity image GANs (hinge + R1 + spectral/equalized-lr).
- **Neural audio codecs** — multi-scale/multi-period discriminators (HiFi-GAN, EnCodec, DAC) for waveform fidelity.

## Common mistake

Blaming "the GAN loss" for instability when the failure is almost always (a) an unconstrained discriminator overpowering G — fix with spectral norm / R1, not a fancier loss; or (b) using the saturating min log(1−D) form and seeing dead gradients. The choice between hinge and non-saturating matters far less than the Lipschitz/gradient control on D.

## See also
- [[vq-vae-discrete-visual-tokenizers]] — VQGAN adds this adversarial term to its reconstruction objective
- [[diffusion-distillation]] — modern few-step distillation reuses adversarial losses to restore sharpness
- [[maximum-mean-discrepancy-wasserstein-distance]] — Wasserstein distance is the WGAN reformulation of the same generative-matching goal
