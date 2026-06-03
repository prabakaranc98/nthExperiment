# Classifier-Free Guidance (CFG)

**One-liner:** Jointly train conditional and unconditional models via random condition-dropout, then at sampling extrapolate the conditional prediction away from the unconditional one to trade diversity for prompt fidelity — the default conditioning knob in diffusion/flow generation.

## The formula

Train one network for both ε_θ(x_t, t, c) and the unconditional ε_θ(x_t, t, ∅), dropping the condition c → ∅ with probability ~10-20% during training. At sampling, replace the model output with the guided prediction:

ε̃ = ε_uncond + w·(ε_cond − ε_uncond) = (1+w)·ε_cond − w·ε_uncond

- w = 0 → unconditional; w = 1 (sometimes called scale 1, no guidance) → plain conditional; w > 1 → amplified conditioning.
- Note convention clash: some define guidance scale s = 1+w, so "scale 7.5" means w = 6.5. Always check which.
- For score/flow models the same linear combination applies to the predicted score (or velocity in flow matching): v_uncond + w·(v_cond − v_uncond).

CFG approximates sampling from p(x|c) ∝ p(x)·p(c|x)^w — a sharpened (tempered) conditional, replacing the explicit classifier gradient of classifier guidance with the implicit unconditional model.

## Where it appears

- **Stable Diffusion / SDXL / Imagen** — text-to-image; scale ~5-12 is the single biggest quality/adherence dial.
- **DiT, latent diffusion, EDM** — same trick on transformer/score-parameterized backbones.
- **Flow matching (SD3, Flux)** — applied to the velocity field; Flux Schnell/Dev distill the guided trajectory.
- **Autoregressive & masked image generation (MAR, MaskGIT-style, LlamaGen)** — CFG on token logits.
- **Distillation (LCM, guidance distillation)** — bake a fixed w into a student so one forward = guided output.

## Common mistake

Treating large w as free quality. High guidance over-saturates colors, blows out contrast, and collapses diversity (mode-seeking); it samples from a sharpened distribution, not the true conditional. Fixes in practice: dynamic/CFG rescaling (Lin et al., "common diffusion noise schedules"), limited-interval guidance (apply only at mid timesteps), or guidance scheduling rather than a single constant.

## See also
- [[ddpm]] — the diffusion sampling loop CFG modifies at each denoising step
- [[flow-matching]] — CFG transfers directly to the velocity field
- [[diffusion-distillation]] — distills the guided trajectory into a single fixed-scale student
