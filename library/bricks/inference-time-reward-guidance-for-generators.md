# Inference-Time / Reward Guidance for Generators

**One-liner:** Steer a (usually diffusion/flow) generator toward a reward r(x) — at sampling time via reward-gradient guidance or best-of-N, or by fine-tuning the sampler with RL (DDPO) or backprop-through-sampling (DRaFT) — the generative analog of RLHF.

## The four families

**1. Reward-gradient (classifier) guidance — no training.** At each denoising step, nudge the predicted mean by the reward gradient w.r.t. the current latent:
  x̃₀ = predict_x0(xₜ, t);  xₜ ← xₜ + s·σₜ²·∇_{xₜ} r(x̃₀)
Reconstruction-guidance / Universal Guidance / FreeDoM apply r to the one-step x̂₀ estimate. Cheap, training-free, but reward must be (sub)differentiable on noisy/decoded samples and steers off-manifold if s too large.

**2. RL fine-tuning (DDPO / DPOK).** Treat the T-step denoising chain as an MDP (state = (xₜ,t), action = xₜ₋₁) and optimize E[r(x₀)] with policy gradient on the per-step Gaussian log-prob:
  ∇θ J = E[ Σₜ r(x₀) ∇θ log pθ(xₜ₋₁ | xₜ, t) ]   (DDPO uses a PPO-style clipped importance ratio).
Works with non-differentiable r (black-box scorers, human prefs).

**3. Backprop-through-sampling (DRaFT / AlignProp / ReFL).** Make r differentiable end-to-end through the sampler and do direct gradient ascent: ∇θ r(sample(θ)). DRaFT-K backprops only through the last K steps (and DRaFT-LV) to cut memory; far more sample-efficient than RL but needs differentiable r and risks reward over-optimization fast.

**4. Best-of-N / search — no training.** Sample N, return argmaxᵢ r(xᵢ). Generative counterpart of rejection sampling; FK-steering / SMC / twisted-SVDD resample particles mid-trajectory using intermediate reward as proposal weights (sequential Monte Carlo over the diffusion chain).

## Where it appears

- **DDPO** (Black et al. 2023) — RL fine-tunes Stable Diffusion for compressibility, aesthetics, prompt-image alignment via a VLM.
- **DRaFT / AlignProp / ReFL** (2023–24) — backprop reward (LAION-Aesthetic, ImageReward, PickScore) through the last few sampling steps.
- **Diffusion-DPO** (2023) — preference optimization on image pairs; the diffusion analog of DPO, no explicit reward model.
- **FK / TDS / SMC steering** (2024–25) — inference-time particle/feynman-kac steering for text and image diffusion, including discrete/masked diffusion LMs.
- **Inference-time alignment of LLMs** — best-of-N and reward-guided decoding (controlled/value-guided decoding) reuse the same logic on autoregressive samplers.

## Common mistake

Applying the reward to the *noisy* latent xₜ instead of the denoised estimate x̂₀(xₜ,t). The reward model was trained on clean data; scoring xₜ directly gives meaningless gradients. Always guide via the predicted x̂₀. The other classic error: cranking guidance scale / over-training on the reward, which collapses diversity and exploits the reward model (reward hacking) — quality on the proxy soars while true quality drops.

## See also
- [[classifier-free-guidance]] — the conditioning-strength knob reward guidance generalizes
- [[rlhf]] — same reward-maximization paradigm for autoregressive policies
- [[reward-hacking-over-optimization]] — the failure mode all four families share
