# RLHF

**One-liner:** Align a model to human preferences via three stages — SFT, then a reward model trained on pairwise comparisons, then PPO that maximizes reward minus a per-token KL penalty to the frozen reference policy.

## The pipeline

**1. SFT.** Fine-tune the base model on curated demonstrations to get π_ref (the reference / initial policy).

**2. Reward model.** Collect comparisons (prompt x, responses y_w ≻ y_l). Train scalar reward r_φ under the Bradley–Terry model:

  L(φ) = −E[ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]

r_φ is π_ref with the LM head swapped for a scalar head; reward is read off the final token.

**3. PPO.** Optimize the policy π_θ to maximize the KL-regularized objective:

  max_θ  E_{x, y∼π_θ}[ r_φ(x,y) ] − β · KL( π_θ(·|x) ‖ π_ref(·|x) )

In practice the KL is folded into a per-token reward: R_t = r_φ·𝟙[t=end] − β·(log π_θ − log π_ref), then PPO maximizes the clipped surrogate with GAE advantages and a value head.

## Where it appears

- InstructGPT / ChatGPT (Ouyang et al. 2022) — the original recipe that made instruction following work.
- Llama 2 / 3 — RLHF with two reward models (helpfulness + safety); rejection sampling + PPO.
- Constitutional AI / RLAIF (Anthropic) — AI-generated preferences replace human labels for the RM.

## Common mistake

Treating the KL penalty as optional regularization. It is load-bearing: without it the policy "reward-hacks" — drifting into out-of-distribution text that scores high under r_φ but is degenerate (the RM is only accurate near π_ref). β sets the exploit-vs-stay-close tradeoff, not just smoothness.

## See also
- [[dpo]] — closed-form policy = implicit reward, skips the RM + PPO loop entirely
- [[grpo]] — drops the value net, normalizes rewards within a sampled group
- [[kl-divergence]] — the penalty anchoring π_θ to π_ref
