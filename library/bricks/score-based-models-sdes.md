# Score-Based Models / SDEs

**One-liner:** Generation as learning the score ∇ₓ log p(x) and integrating a reverse-time SDE (or the equivalent probability-flow ODE); the continuous-time framework that unifies DDPM (VP-SDE) and SMLD (VE-SDE) into one object (Song et al., 2021).

## The formula

Forward (data → noise) SDE adds noise over t ∈ [0,1]:

dx = f(x,t) dt + g(t) dw

Anderson's reverse-time SDE (run t: 1 → 0) recovers the data:

dx = [f(x,t) − g(t)² ∇ₓ log pₜ(x)] dt + g(t) dw̄

The **only unknown** is the score ∇ₓ log pₜ(x). Learn it by denoising score matching:

L = 𝔼ₜ,ₓ₀,ₓₜ [ λ(t) ‖ sθ(xₜ,t) − ∇ₓₜ log p(xₜ|x₀) ‖² ]

For Gaussian perturbation kernels the target is −(xₜ−mean)/σₜ², i.e. score matching = predicting the noise ε up to scaling.

**Probability-flow ODE** — same marginals pₜ(x), deterministic, exactly invertible:

dx = [f(x,t) − ½ g(t)² ∇ₓ log pₜ(x)] dt

## The two regimes

- **VP-SDE** (variance preserving): f = −½β(t)x, g = √β(t) → continuous limit of DDPM; variance stays bounded.
- **VE-SDE** (variance exploding): f = 0, g = √(d[σ²(t)]/dt) → continuous limit of SMLD/NCSN; variance grows unbounded.

## Where it appears

- **DDPM / SMLD** — both are discretizations of the VP- and VE-SDE; "predict noise" ≡ "predict score".
- **EDM (Karras et al., 2022)** — recasts everything as score + ODE, then engineers preconditioning, σ-schedule, and Heun/2nd-order ODE samplers for SOTA sampling.
- **Consistency / distillation models** — distill the PF-ODE trajectory into 1–4 step samplers.
- **Stable Diffusion / latent diffusion / DiT** — the score net (now a Transformer) operates in VAE latent space; samplers (DDIM, DPM-Solver) are PF-ODE solvers.
- **Classifier(-free) guidance** — adds a guidance term to the learned score to steer conditional generation.
- **Flow matching / rectified flow** — same reverse-process idea but learns the velocity field directly instead of the score (linearly related).

## Common mistake

Conflating the reverse SDE with the probability-flow ODE. They share the same per-timestep marginals pₜ(x) but produce different sample paths: the SDE injects fresh noise (more diverse, robust to score error), the ODE is deterministic and invertible (enables exact likelihoods, latents, and few-step distillation). Picking the wrong one — or mismatching the discretization to the SDE family — is where samplers silently degrade.

## See also
- [[ddpm]] — the discrete VP-SDE special case; "predict ε" = "predict score"
- [[flow-matching]] — learns the velocity field instead of the score, same reverse-time generation
- [[edm-preconditioning-ode-samplers]] — the score+ODE reformulation that made sampling fast and clean
