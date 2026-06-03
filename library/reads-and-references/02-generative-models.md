# Generative Models

From VAEs and GANs through diffusion models and flow matching — the conceptual and mathematical resources for understanding generative modeling at the frontier.

---

## The diffusion + score matching canon (Lilian Weng)

Lilian Weng has written the definitive survey-level posts on diffusion and score-based models. Start here before reading the original papers.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | The complete survey: DDPM, score matching, SDEs, DDIM, classifier guidance — all in one post with derivations. The canonical reference before reading Ho/Song papers. | 🟡 |
| [Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) | How diffusion extends to video — temporal modeling, action conditioning, world model connections. Current frontier. | 🟡 |

---

## Illustrated guides (Jalammar)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | Latent diffusion step by step — text encoder, VAE, U-Net, the full Stable Diffusion pipeline. Best visual introduction. | 🟢 |

---

## Distill: GANs and the generative landscape

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Open Questions about GANs](https://distill.pub/2019/gan-open-problems/) | The unsolved problems in GAN training as of 2019 — still relevant for understanding why diffusion won. | 🟡 |
| [Differentiable Image Parameterizations](https://distill.pub/2018/differentiable-parameterizations/) | How differentiable rendering and image generation connect — the link between generative models and neural network visualization. | 🟡 |

---

## Flow matching (new primary resource)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Flow Matching and Diffusion — Tutorial](https://arxiv.org/abs/2506.02070) | The 2025 tutorial unifying diffusion and flow matching under generator matching. Essential for understanding the current generative frontier. | 🔴 |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | Straight-line transport between distributions — the conceptual shift that enables 1-step generation. | 🔴 |

---

## Understanding VAEs and the ELBO

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Colah: Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | The information-theoretic foundations that make the VAE ELBO derivation legible — entropy, KL, coding theory. | 🟢 |
| [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) | VAE derivation, the reparameterization trick, β-VAE for disentanglement — the complete survey. | 🟡 |

---

## Controllable generation

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng: Controllable Neural Text Generation](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/) | How to steer generation: prompting, fine-tuning, PPLM, classifier guidance — the historical arc. | 🟡 |
| [Distill: Feature-wise Transformations (FiLM)](https://distill.pub/2018/feature-wise-transformations/) | The general conditioning framework behind classifier-free guidance and many controllable models. | 🟡 |
