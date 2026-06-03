# DDPM (Denoising Diffusion)

**One-liner:** A forward process gradually corrupts data into Gaussian noise over T steps; a network is trained to predict that noise, and sampling reverses the corruption by iterative denoising from pure noise.

## The formula / definition

**Forward (fixed, no learning):** add Gaussian noise on a variance schedule β₁..β_T.
- q(xₜ | x_{t-1}) = N(xₜ; √(1-βₜ) x_{t-1}, βₜ I)
- Closed form (the key trick): with αₜ = 1-βₜ and ᾱₜ = ∏_{s≤t} αₛ,
  **xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε,  ε ~ N(0, I)** — jump to any t in one step.

**Reverse (learned):** p_θ(x_{t-1} | xₜ) = N(x_{t-1}; μ_θ(xₜ, t), Σₜ). Ho et al. (2020) reparameterize μ_θ via a noise-prediction net ε_θ.

**Training loss (the simple objective):** sample t, x₀, ε; form xₜ; regress the noise:
  **L = E_{t,x₀,ε} ‖ ε − ε_θ(√ᾱₜ x₀ + √(1-ᾱₜ) ε, t) ‖²**
(a reweighted ELBO; the constant weighting works better than the variational weights.)

**Sampling (ancestral):** x_T ~ N(0,I); for t=T..1:
  x_{t-1} = (1/√αₜ)(xₜ − (βₜ/√(1-ᾱₜ)) ε_θ(xₜ,t)) + σₜ z,  z~N(0,I) (z=0 at t=1).

## Where it appears

- **DDPM (Ho et al. 2020)** — the canonical formulation above; ε-prediction + simplified L².
- **DDIM (Song et al. 2020)** — non-Markovian, deterministic sampler; same trained net, 20–50 steps instead of 1000.
- **Latent Diffusion / Stable Diffusion, SDXL** — run DDPM in a VAE latent space; the workhorse of open image generation.
- **Classifier-free guidance** — jointly train conditional/unconditional ε_θ, extrapolate at sample time for prompt adherence.
- **v-prediction / EDM (Karras 2022)** — better parameterization and noise schedules; standard in modern training.

## Common mistake

Thinking the network "predicts the clean image" or "the previous frame." Standard DDPM predicts the **noise ε** added to x₀ (equivalently x₀ or v — interconvertible). Also: the forward process is *fixed and parameter-free*; only the reverse is learned. And T steps in training ≠ T steps in sampling — DDIM/distillation cut sampling steps drastically without retraining.

## See also
- [[flow-matching]] — continuous-time ODE alternative; subsumes diffusion as a special probability path
- [[reparameterization]] — the √ᾱₜ x₀ + √(1-ᾱₜ) ε trick is exactly reparameterized sampling
- [[kl-divergence]] — the training objective is a KL-based ELBO between forward and reverse posteriors
