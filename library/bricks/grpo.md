# GRPO — Group Relative Policy Optimization

**One-liner:** Critic-free RL for LLMs — estimate the advantage baseline from a group of sampled responses instead of a value network; combined with verifiable rewards, this is the DeepSeek-R1 recipe.

## The algorithm

For a prompt x, sample G responses {o₁, ..., o_G} from the current policy π_θ:

1. Compute reward rᵢ for each response (e.g., 1 if correct, 0 if wrong)
2. Compute **group-relative advantage**: Aᵢ = (rᵢ − mean(r)) / std(r)
3. Update with PPO-style clipped objective using Aᵢ:
   - rₜ(θ) = π_θ(oᵢ|x) / π_θ_old(oᵢ|x)
   - L = E[min(rₜ·Aᵢ, clip(rₜ, 1−ε, 1+ε)·Aᵢ)]

**No critic network needed** — the group itself provides the baseline.

## Why no critic?

PPO needs a value function V(s) to estimate the baseline. For LLMs, the "state" is the full prompt + partial generation — a critic must be as large as the policy to understand it. GRPO uses the group mean as a cheap, unbiased baseline estimate.

## Where it appears

- **DeepSeekMath** ([arXiv 2402.03300](https://arxiv.org/abs/2402.03300)) — original GRPO for math reasoning
- **DeepSeek-R1** ([arXiv 2501.12948](https://arxiv.org/abs/2501.12948)) — reasoning emergence from pure RL with GRPO
- **Tülu 3 / RLVR** — GRPO with verifiable rewards (code, math)
- **DAPO** — decoupled-clip + dynamic-sampling extension of GRPO

## The key pairing: GRPO + verifiable rewards (RLVR)

GRPO works best when rewards are binary and verifiable (right/wrong). With a free, unhackable reward signal, you don't need a reward model, and GRPO provides a cheap advantage estimator. This is the 2025 post-training recipe.

## Common mistake

Using GRPO with noisy/learned reward models defeats its purpose. The advantage estimate from a noisy reward is noisy too — you lose the stability benefit. GRPO shines specifically with clean, verifiable rewards.

## See also
- [[dpo]] — offline preference alternative; no RL needed
- [[log-derivative-trick]] — the mathematical foundation of policy gradients
- [[scaling-laws]] — reasoning RL improves models in ways that scale similarly to pretraining
