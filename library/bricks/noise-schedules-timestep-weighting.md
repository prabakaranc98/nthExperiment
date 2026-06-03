# Noise Schedules & Timestep Weighting

**One-liner:** The β/σ schedule, SNR parameterization, timestep sampling distribution, and resolution-dependent shift jointly decide *which noise levels the model actually trains on* — i.e. where capacity gets spent; min-SNR weighting and cosine/shifted schedules are the recurring fixes.

## The formula / definition

A forward process noises x₀: x_t = √(ᾱ_t) x₀ + √(1−ᾱ_t) ε. Define **log-SNR** λ_t = log(ᾱ_t / (1−ᾱ_t)). Everything is reparameterizable through λ; the "schedule" is just the map t → λ_t and the "weighting" is the per-t loss multiplier.

- **Linear β (DDPM):** β_t ramps 1e-4→0.02. Spends too much budget at high SNR (nearly-clean), wastes steps.
- **Cosine (Nichol & Dhariwal):** ᾱ_t = cos²((t/T+s)/(1+s)·π/2). Smoother, more mass at mid-noise.
- **EDM (Karras):** parameterize directly in σ; train σ ~ lognormal (P_mean=-1.2, P_std=1.2). Preconditioning c_skip/c_out/c_in make the network see unit-variance inputs.
- **min-SNR-γ weighting:** scale the ε-loss at step t by min(SNR(t), γ) (γ≈5). Caps the dominance of low-noise steps → treats denoising as multi-task with balanced tasks.
- **Logit-normal sampling (SD3 / flow matching):** sample t with density concentrated near 0.5; π(t) ∝ (1/(t(1−t)))·N(logit(t); m, s²).
- **Resolution shift:** at higher resolution, a fixed σ corrupts *less* perceptual signal (redundant pixels). Shift the schedule: SNR_shifted = SNR · (ref_res / res)². SD3 shifts log-SNR by log(m); higher-res ⇒ push toward more noise.

## Where it appears

- **Stable Diffusion 3 / flow matching** — logit-normal t-sampling + resolution-dependent timestep shift, the key recipe change vs. SDXL.
- **EDM / EDM2 (Karras)** — σ-space lognormal sampling + preconditioning; defines the modern noise-level training distribution.
- **min-SNR-γ (Hang et al.)** — drop-in loss reweighting that ~3.4× speeds up DiT/U-Net convergence.
- **Imagen / eDiff-I / DiT** — cosine or shifted schedules; cascades use different schedules per resolution stage.
- **Consistency / distillation** — the teacher's schedule fixes the ODE the student must match.

## Common mistake

Conflating the **schedule** (forward noising map t→σ) with the **timestep weighting** (loss multiplier) and the **sampling distribution** (which t you draw during training). They are independent knobs: ε-prediction, v-prediction, and x₀-prediction give *identical* SNR-weighted objectives but *different* implicit weightings, so changing the parameterization silently re-weights training. Also: a schedule tuned at 256px does not transfer to 1024px without the resolution shift — the model trains on effectively-too-clean inputs and the top SNR never sees enough noise.

## See also
- [[ddpm]] — defines the β schedule and ε-objective these weightings reparameterize
- [[flow-matching]] — the SD3/logit-normal formulation lives here; SNR ↔ flow time mapping
- [[edm-preconditioning-ode-samplers]] — σ-space lognormal sampling and input preconditioning
