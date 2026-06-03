# Generalized Advantage Estimation (GAE)

**One-liner:** λ-weighted exponential average of multi-step TD residuals that estimates the advantage A(s,a) with a tunable bias-variance knob — the "A" that PPO's clipped surrogate multiplies against the policy ratio.

## The formula

Define the one-step TD residual using a learned value function V:

δ_t = r_t + γ V(s_{t+1}) − V(s_t)

GAE is the discounted, λ-weighted sum of future residuals:

Â_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

Computed efficiently backward over a rollout:

Â_t = δ_t + γλ Â_{t+1}

The value target is then R_t = Â_t + V(s_t) (used to regress V).

## The bias-variance knob

- λ = 0  ⇒  Â_t = δ_t = r_t + γV(s_{t+1}) − V(s_t): pure one-step TD, low variance, high bias.
- λ = 1  ⇒  Â_t = Σ γ^l r_{t+l} − V(s_t): full Monte-Carlo return minus baseline, unbiased, high variance.
- Typical: γ ≈ 0.99, λ ≈ 0.95.

## Where it appears

- **PPO** (Schulman et al., 2017) — the canonical advantage estimator plugged into the clipped surrogate; almost always γ=0.99, λ=0.95.
- **RLHF / InstructGPT** — PPO over a reward model uses GAE; the value head is a second head on the policy.
- **GRPO** (DeepSeek) — explicitly *drops* GAE and the value network, replacing A with group-normalized rewards (mean/std over k samples) to avoid training a critic.
- **RLVR / reasoning RL** — many verifier-reward setups for long CoT keep PPO+GAE; others move to GRPO-style baselines precisely to skip the critic.

## Common mistake

Confusing the two discount-like parameters. γ discounts *rewards* (defines the objective). λ only controls the *bias-variance tradeoff of the estimator* — it does not change what is being optimized. Also: GAE needs a learned V(s); without a critic (as in GRPO) you are not doing GAE, you are using a different baseline.

## See also
- [[ppo-clipped-surrogate-objective]] — consumes Â_t as the advantage term in the ratio-clipped loss
- [[grpo]] — drops GAE/critic in favor of group-relative reward baselines
- [[gradient-variance-reduction-control-variates]] — V(s_t) is the baseline / control variate that GAE subtracts
