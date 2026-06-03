# Consistency Models

**One-liner:** A model f_theta(x_t, t) trained to map any point on a probability-flow ODE trajectory directly to that trajectory's origin x_0, enabling 1-step generation while still supporting iterative multi-step refinement (CM, LCM, sCM).

## The formula / definition

Define the PF-ODE whose solution flows noisy x_t to clean x_0. The consistency function must satisfy:
- **Boundary condition:** f_theta(x, epsilon) = x at the smallest time epsilon (parameterized via skip connection so it holds by construction): f_theta = c_skip(t)*x + c_out(t)*F_theta(x,t).
- **Self-consistency:** f_theta(x_t, t) = f_theta(x_{t'}, t') for all t, t' on the same trajectory.

Training enforces consistency between adjacent timesteps:
L = E[ d( f_theta(x_{t_{n+1}}, t_{n+1}), f_theta-(x_{t_n}, t_n) ) ]
where x_{t_n} is one ODE solver step back from x_{t_{n+1}}, theta- is a target net (EMA or, in sCM, stop-grad), and d is L2 / LPIPS / Pseudo-Huber. **Distillation (CD)** uses a pretrained score model for the ODE step; **isolation (CT)** estimates the step from data + noise, training from scratch. Sample in 1 step: x_0 ~= f_theta(z, T); or multi-step: alternate denoise + re-noise.

## Where it appears

- **Consistency Models (Song et al. 2023)** — original CD/CT; few-step image generation from diffusion.
- **Latent Consistency Models (LCM) / LCM-LoRA** — consistency distillation in SD latent space; 2-4 step SDXL, distilled as a portable LoRA.
- **sCM / TrigFlow (Lu & Song 2024)** — continuous-time consistency (no discretization), stabilized parameterization closing the quality gap to diffusion with 2 steps.
- **Real-time / streaming generation** — LCM and turbo-style models power interactive image and video pipelines.

## Common mistake

Conflating the two training modes. **Consistency distillation** needs a pretrained diffusion teacher to take the ODE step; **consistency training** does not and learns the consistency function directly from data. Also: the target network theta- is not optional decoration in discrete CT — it (or stop-grad in sCM) is what prevents the trivial collapse to a constant function.

## See also
- [[diffusion-distillation]] — consistency distillation is one family within few-step distillation
- [[score-based-models-sdes]] — the PF-ODE/score model that defines the trajectories
- [[edm-preconditioning-ode-samplers]] — the c_skip/c_out preconditioning and ODE solver borrowed by CMs
