# Gradient Variance Reduction & Control Variates

**One-liner:** Subtract a correlated, zero-mean term (baseline / control variate) from a Monte Carlo gradient estimator to slash its variance without adding bias — the engineering that makes REINFORCE usable and the formal reason GRPO's group-mean baseline works.

## The formula / definition

A **control variate** replaces estimator ĝ with ĝ − c·(h − E[h]). Since E[h − E[h]] = 0, the estimator stays unbiased; variance drops by 2c·Cov(ĝ,h) − c²Var(h), minimized at c* = Cov(ĝ,h)/Var(h), giving variance reduction ∝ ρ² (squared correlation between ĝ and h).

In policy gradients the canonical control variate is an **action-independent baseline** b(s):

∇θ J = E[(R − b(s))·∇θ log π_θ(a|s)]

This is unbiased for *any* b not depending on the sampled action a, because E[b(s)·∇θ log π_θ(a|s)] = b(s)·∇θ Σ_a π_θ = b(s)·∇θ 1 = 0. The variance-minimizing baseline is the score-weighted expected return b*(s) = E[(∇log π)²·R] / E[(∇log π)²] ≈ V(s); in practice one uses V(s), the batch mean, or the group mean.

## Where it appears

- **REINFORCE with baseline** (Williams 1992) — subtract V(s) or a running reward mean; without it the estimator is too noisy to train.
- **Actor-critic / GAE** — learned V(s) is the baseline; advantage A = R − V is the control-variated signal. PPO optimizes A, not raw R.
- **GRPO** (DeepSeek-R1) — baseline = group mean of rewards over G sampled completions; (r − mean)/std is a critic-free control variate, exploiting that all G samples share the same prompt/state.
- **VAE / reparameterization fallbacks** — RELAX, REBAR, NVIL use learned/Taylor control variates for discrete-latent gradients where reparameterization fails.
- **RL rollout systems (2024–2026)** — leave-one-out baselines (RLOO) and group baselines are standard for verifiable-reward LLM RL.

## Common mistake

Making the baseline depend on the *sampled action* (or otherwise correlated with the noise it's meant to cancel) — this breaks the E[b·∇log π]=0 identity and introduces **bias**. The baseline may depend on the state/prompt freely, but never on the action drawn. A baseline only reduces variance; it never changes the expected gradient.

## See also
- [[log-derivative-trick]] — the high-variance estimator that baselines are bolted onto
- [[grpo]] — group-mean baseline as a critic-free control variate
- [[generalized-advantage-estimation]] — bias–variance knob (λ) on the advantage control variate
