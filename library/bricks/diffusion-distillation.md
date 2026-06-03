# Diffusion Distillation (Few-Step)

**One-liner:** Compress a slow many-step (50-1000 NFE) diffusion teacher into a 1-8 step student via progressive halving, consistency self-distillation, adversarial (ADD/LADD), or distribution-matching (DMD/DMD2) objectives — the lever that turns diffusion into real-time generation.

## The four families

**Progressive distillation (Salimans & Ho 2022):** student learns to take one DDIM step that equals two teacher steps; iterate, halving NFE each round (1024 → 512 → ... → 4). Target = two-step teacher ODE update; ~log2(steps) distillation rounds.

**Consistency distillation (Song 2023):** train f_θ so any point on the same PF-ODE trajectory maps to the same origin x_0. Loss = d( f_θ(x_t, t), f_θ⁻(x_{t'}, t') ) along teacher-ODE-adjacent pairs, θ⁻ an EMA target. Enables 1-2 step sampling. LCM applies this in latent space; sCM/TrigFlow (2024) stabilizes continuous-time training.

**Adversarial (ADD / LADD):** add a GAN discriminator on top of (or instead of) the distillation loss so few-step samples are sharp, not blurry-averaged. ADD (SDXL-Turbo) discriminates in pixel space + score-distillation term; LADD (SD3-Turbo) moves the discriminator into latent features of the diffusion model itself.

**Distribution matching (DMD / DMD2):** minimize approximate reverse-KL between student-output distribution and teacher distribution. Gradient uses two score networks:
  ∇θ KL ≈ E[ (s_fake(x) − s_real(x)) · ∂x/∂θ ]
where s_real is the frozen teacher score and s_fake is a score model trained online on student samples. DMD2 (2024) drops the costly regression loss, adds a GAN term, and supports multi-step → near-teacher quality at 1-4 NFE.

## Where it appears

- SDXL-Turbo / SD3-Turbo — ADD/LADD for 1-4 step real-time image generation
- LCM / LCM-LoRA — consistency distillation as a plug-in LoRA on any SD checkpoint
- DMD2 — 1-step generators matching multi-step SDXL on FID; basis for many 2024-25 fast samplers
- FLUX/SD3 "schnell"/turbo variants, fast video (e.g. distilled SVD), and audio diffusion serving

## Common mistake

Conflating few-step *distillation* with few-step *samplers* (DPM-Solver, EDM heun). Better ODE solvers reduce NFE on the *same* model with no retraining but plateau around 10-20 steps; distillation retrains weights to collapse the trajectory and reaches 1-4 steps — at the cost of mode coverage/diversity (reverse-KL is mode-seeking, which is why GAN/multi-step terms are added back).

## See also
- [[consistency-models]] — the self-distillation objective behind LCM and 1-step sampling
- [[edm-preconditioning-ode-samplers]] — the teacher PF-ODE and solver baseline distillation competes with
- [[gan-objective-adversarial-losses]] — the adversarial term in ADD/LADD/DMD2 that restores sharpness
