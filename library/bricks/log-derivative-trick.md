# Log-Derivative Trick (REINFORCE / Score Function)

**One-liner:** ∇θ E_{x∼p_θ}[f(x)] = E_{x∼p_θ}[f(x)·∇θ log p_θ(x)] — turns the gradient of an expectation into an expectation of a gradient you can Monte Carlo estimate, even when f is non-differentiable or x is discrete; the basis of policy gradients.

## The derivation

The parameters live inside the sampling distribution, so you can't push ∇θ through the expectation directly. Use the identity ∇θ p_θ = p_θ · ∇θ log p_θ:

∇θ E[f] = ∇θ ∫ f(x) p_θ(x) dx = ∫ f(x) ∇θ p_θ(x) dx
        = ∫ f(x) p_θ(x) ∇θ log p_θ(x) dx = E_{x∼p_θ}[ f(x) ∇θ log p_θ(x) ]

The term ∇θ log p_θ(x) is the **score function**. Monte Carlo estimate: (1/N) Σ f(xᵢ) ∇θ log p_θ(xᵢ), xᵢ ∼ p_θ. Only requires evaluating f (a black box reward) and differentiating log-prob — never differentiating f itself.

## Variance reduction (essential in practice)

Raw estimator has huge variance. Subtract a **baseline** b that doesn't depend on the sampled action: E[(f(x) − b)·∇θ log p_θ] is unbiased because E[b·∇θ log p_θ] = b·∇θ E[1] = 0. Use b = value baseline (advantage A = R − V), batch-mean reward, or group-mean (GRPO).

## Where it appears

- **REINFORCE / vanilla policy gradient** (Williams 1992) — ∇θ J = E[ Σ ∇θ log π_θ(aₜ|sₜ) · Rₜ ]; f = return, p_θ = policy.
- **PPO / RLHF** — the policy-gradient core; PPO clips the importance ratio but the underlying estimator is score-function based.
- **GRPO** (DeepSeek) — group-relative baseline replaces a learned value net; advantage = (r − group mean)/group std, plugged into the log-derivative estimator.
- **Discrete latent / hard attention / NAS** — when reparameterization is impossible because samples are discrete.
- **Black-box / evolutionary gradient estimation** — score-function estimator over parameter-perturbation distributions.

## Common mistake

Thinking you need ∇f. You don't — f can be a non-differentiable, discrete, simulator, or human reward; only log p_θ must be differentiable. The dual mistake: forgetting that a valid baseline must be **independent of the sampled action** (or correlated only through the state), otherwise subtracting it introduces bias.

## See also
- [[reparameterization]] — the lower-variance alternative when x is a differentiable transform of θ and noise (continuous, pathwise gradients)
- [[grpo]] — score-function policy gradient with a group-relative baseline
- [[rlhf]] — policy optimization built on this estimator
