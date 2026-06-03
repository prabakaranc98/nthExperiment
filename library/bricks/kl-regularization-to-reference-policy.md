# KL Regularization to Reference Policy

**One-liner:** Penalize divergence from the frozen SFT/reference model (β·KL[π_θ‖π_ref]) during RL so the policy chases reward without collapsing into degenerate, off-distribution text; the central stability knob across PPO, DPO, and GRPO.

## The formula / definition

The KL-regularized RL objective (RLHF):

max_θ  E_{x∼D, y∼π_θ}[ r(x,y) ] − β·KL[π_θ(y|x) ‖ π_ref(y|x)]

This has a known closed-form optimum: π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β). This identity is exactly what DPO inverts to express the reward implicitly via log-prob ratios.

In PPO the KL is folded into a per-token reward shaping term:
  r̃_t = r(x,y)·1[t=T] − β·log( π_θ(a_t|s_t) / π_ref(a_t|s_t) )
i.e. the KL penalty appears as a per-token negative reward, not a separate loss. Practitioners often use the **k3 unbiased estimator** of KL: (ρ − 1) − log ρ where ρ = π_ref/π_θ, which is always ≥ 0 and lower-variance than naive −log ρ.

## Where it appears

- **InstructGPT / RLHF** — adds β·KL to PPO reward to keep the policy near the SFT model; β is the primary defense against reward hacking.
- **DPO** — the β·KL term is *baked into* the loss derivation; β controls how aggressively the implicit reward separates chosen vs. rejected. Higher β = stay closer to π_ref.
- **GRPO (DeepSeek-Math/R1)** — keeps an explicit KL term (k3 estimator) added directly to the loss rather than reward; R1-Zero/some recipes drop it entirely for pure RLVR to allow large distribution shift.
- **RLAIF / Constitutional AI** — same anchor against a reference policy when reward comes from an AI judge.

## Common mistake

Treating β as a fixed safe constant and ignoring that the *reference* itself defines what "degenerate" means. If π_ref is weak, a large β chains the policy to a bad prior; if β is too small, the policy reward-hacks and drifts into high-reward gibberish the reward model overrates. Also: per-token KL in PPO is reward shaping, not a constraint — it does not hard-bound total divergence the way a trust region would.

## See also
- [[kl-divergence]] — the underlying asymmetric divergence being penalized
- [[dpo]] — folds this exact β·KL term into a closed-form supervised loss
- [[reward-hacking-over-optimization]] — what insufficient KL regularization lets happen
