# EDM Preconditioning & ODE Samplers

**One-liner:** Karras et al. (2022) reframe diffusion as a single noise-level (sigma) continuum with c_skip/c_out/c_in/c_noise network preconditioning, a log-normal training schedule, and tuned ODE/SDE samplers (Heun, DPM-Solver++) — cleanly decoupling training from sampling and now the de facto design space.

## The preconditioning

Train a network F_theta but parameterize the denoiser D as a skip-weighted combination so the network input/output stay unit-variance at every sigma:

D_theta(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))

With sigma_data the data std:
- c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
- c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
- c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
- c_noise = ln(sigma) / 4   (the network sees log-sigma, not t)

Loss: E[ lambda(sigma) * c_out^2 * || F_theta(...) - target ||^2 ] with weighting that makes the effective loss ~uniform across sigma. Training samples ln(sigma) ~ N(P_mean, P_std^2) (defaults P_mean=-1.2, P_std=1.2).

## Sampling: the probability-flow ODE

dx = -sigma * score(x; sigma) dsigma, and score = (D_theta(x; sigma) - x) / sigma^2. Discretize sigma on a tuned schedule sigma_i = (sigma_max^{1/rho} + i/(N-1)*(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho with rho=7. Integrate with Heun (2nd-order, one correction step), optionally adding stochasticity (the EDM "churn" SDE). DPM-Solver++ / UniPC are higher-order exponential integrators hitting ~10-20 steps.

## Where it appears

- EDM / EDM2 (Karras et al. 2022, 2024) — the formulation itself; EDM2 adds magnitude-preserving layers and post-hoc EMA tuning, SOTA ImageNet FID.
- Consistency Models & distillation — built directly on EDM's sigma-space and preconditioning to learn few-/one-step maps.
- Stable Diffusion 3 / FLUX, Karras-schedule schedulers in diffusers — borrow the sigma schedule and Heun/DPM++ samplers even when trained with flow matching.

## Common mistake

Conflating the training noise distribution with the sampling sigma schedule. They are independent knobs: you train by sampling ln(sigma) from a log-normal, but you sample by stepping a separately tuned rho=7 grid. EDM's whole point is that the optimal sampling schedule has nothing to do with how you drew noise during training.

## See also
- [[score-based-models-sdes]] — EDM is the practical, well-conditioned instantiation of the score SDE/ODE
- [[noise-schedules-timestep-weighting]] — c_skip/c_out and lambda(sigma) are the principled choice of weighting
- [[flow-matching]] — the competing/complementary continuous-time formulation modern samplers share schedulers with
