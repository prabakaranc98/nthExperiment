# Generative Models

From VAEs and GANs through diffusion, score matching, and flow matching — the conceptual and mathematical resources for understanding generative modeling at the frontier. As of mid-2026 the field has converged: diffusion, score-based SDEs, and flow matching are one framework (continuous-time transport of a noise distribution to data), and the practical questions are about transformer backbones, few-step sampling, and the latent space the process runs in.

For the concept-library side (DDPM, latent diffusion, flow matching, DiT, consistency models, VQ-VAE, GAN losses), see [`../bricks/README.md`](../bricks/README.md).

**Difficulty scale:** 🟢 Accessible · 🟡 Intermediate · 🔴 Advanced

---

## Start here

The orientation reads — work through these before opening original papers.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | The complete survey: DDPM, score matching, SDEs, DDIM, classifier guidance — all in one post with derivations. Still the canonical first read. | 🟡 |
| [Yang Song — Score-Based Generative Modeling](https://yang-song.net/blog/2021/score/) | The cleanest narrative for the score / SDE view: estimate ∇ₓ log p(x), perturb across noise levels, sample by reversing the SDE (or its probability-flow ODE). The mental model the whole field now uses. | 🟡 |
| [MIT 6.S184 — An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070) | Self-contained 2025 lecture notes (Holderrieth & Erives) building from ODEs/SDEs to flow matching, score matching, guidance, and modern image/video generators. The best single document for the unified view. | 🔴 |
| [Hugging Face — Diffusion Models Course](https://huggingface.co/learn/diffusion-course/en/unit0/1) | Free, hands-on, notebook-driven: train a model from scratch, fine-tune, run Stable Diffusion, add guidance. The fastest path from theory to running code. | 🟢 |

---

## Key papers

The papers that define how generation is actually done in 2026 — backbone, noise parameterization, and few-step sampling. (Original DDPM and Song et al. score-SDE are covered in depth by the surveys above.)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Elucidating the Design Space of Diffusion Models (EDM)](https://arxiv.org/abs/2206.00364) | Karras et al. untangle diffusion into independent design axes — preconditioning, noise schedule, ODE sampler. The reference recipe almost every modern system inherits. | 🔴 |
| [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) | Replaces the U-Net with a plain transformer over latent patches and shows clean compute-vs-FID scaling. The architecture behind today's frontier image and video generators. | 🔴 |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | Straight-line transport between distributions — the conceptual shift that enables near-1-step generation and underpins modern flow-matching samplers. | 🔴 |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | Map any point on a probability-flow ODE trajectory directly to its origin, enabling high-quality 1-step sampling either by distilling a teacher or training from scratch. | 🔴 |
| [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) | Lipman et al.'s comprehensive, self-contained treatment of flow matching (continuous and discrete) with a companion [PyTorch library](https://github.com/facebookresearch/flow_matching). The definitive flow-matching reference. | 🔴 |

---

## Blogs & explainers

### Sander Dieleman — the diffusion essayist

The best long-form thinking on why diffusion works and where it's going. Read these in order to build a deep mental model.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Perspectives on diffusion](https://sander.ai/2023/07/20/perspectives.html) | Diffusion as autoencoder, latent-variable model, score predictor, reverse-SDE solver, flow, and autoregressive model — all at once. The post that makes every other framing click. | 🟡 |
| [Noise schedules considered harmful](https://sander.ai/2024/06/14/noise-schedules.html) | Why the noise schedule is really about input resolution and the frequency content of your data — and why much schedule folklore is a proxy for that. | 🟡 |
| [Diffusion is spectral autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html) | The sharpest recent intuition: a diffusion model is approximately doing autoregression in the frequency domain, generating coarse-to-fine. Reframes the whole process. | 🟡 |
| [The paradox of diffusion distillation](https://sander.ai/2024/02/28/paradox.html) | Why distilling a many-step teacher into a few-step student works at all, and the trade-offs between distillation families. Background for consistency/few-step models. | 🟡 |
| [Generative modelling in latent space](https://sander.ai/2025/04/15/latents.html) | Why nearly every frontier generator runs in a learned latent space, what the autoencoder buys you, and where it bites back. | 🟡 |
| [Learning the integral of a diffusion model (flow maps)](https://sander.ai/2026/05/06/flow-maps.html) | Flow maps predict any point on a trajectory from any other — generalizing consistency models toward faster sampling and reward-based steering. The current frontier of few-step generation. | 🔴 |

### Surveys & visual guides

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng — Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) | How diffusion extends to video: temporal modeling, action conditioning, the world-model connection. Still the best survey-level overview of the video frontier. | 🟡 |
| [Jalammar — The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | Latent diffusion step by step — text encoder, VAE, U-Net, the full pipeline. The best visual introduction to text-to-image. | 🟢 |
| [Lilian Weng — From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) | VAE derivation, the reparameterization trick, β-VAE for disentanglement — the complete survey. The latent autoencoder is now the substrate under latent diffusion. | 🟡 |
| [Colah — Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | Entropy, KL, and coding theory made geometric — the foundations that make the VAE ELBO and the diffusion variational bound legible. Timeless. | 🟢 |
| [Lilian Weng — Controllable Neural Text Generation](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/) | How to steer generation — prompting, fine-tuning, PPLM, classifier guidance. The conditioning ideas carry directly to diffusion guidance. | 🟡 |
| [Distill — Feature-wise Transformations (FiLM)](https://distill.pub/2018/feature-wise-transformations/) | The general conditioning framework behind classifier-free guidance and adaLN-style conditioning in DiT. Foundational and still relevant. | 🟡 |
| [Distill — Open Questions about GANs](https://distill.pub/2019/gan-open-problems/) | The unsolved problems in GAN training — useful historical context for why diffusion and flow matching displaced GANs as the default. | 🟡 |

---

## Courses & talks

| Resource | What it covers | Level |
|----------|---------------|-------|
| [MIT 6.S184 — Flow Matching and Diffusion Models (2025 lectures)](https://diffusion.csail.mit.edu/) | Full IAP course site: video lectures, labs, and the notes from the "Start here" entry. Generative AI from SDEs through flow matching, taught from first principles. | 🔴 |
| [Hugging Face — Diffusion Models Course](https://huggingface.co/learn/diffusion-course/en/unit0/1) | Four units of theory + notebooks: diffusers basics, fine-tuning and guidance, Stable Diffusion, advanced techniques. The hands-on companion to the math above. | 🟢 |
| [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/) | The later lessons build Stable Diffusion from components. Top-down, code-first — the fastest way to get a generator running and understood. | 🟢 |

---

## Tools

| Resource | What it is | Level |
|----------|-----------|-------|
| [🤗 Diffusers](https://huggingface.co/docs/diffusers/index) | The default library for running, fine-tuning, and composing diffusion/flow pipelines — schedulers, samplers, and pretrained checkpoints in one API. | 🟢 |
| [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | Reference PyTorch implementation of continuous and discrete flow matching, with image and text examples. Pairs with the Flow Matching Guide. | 🟡 |
| [NVlabs/edm](https://github.com/nvlabs/edm) | Official EDM code — the preconditioning, schedules, and ODE samplers from the design-space paper. The cleanest place to study a production-grade diffusion training loop. | 🔴 |
| [openai/consistency_models](https://github.com/openai/consistency_models) | Official consistency-models repo for training and few-step sampling — the entry point for hands-on work on fast generation. | 🔴 |
