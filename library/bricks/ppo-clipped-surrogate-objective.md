# PPO Clipped Surrogate Objective

**One-liner:** Constrain each policy-gradient update with a clipped importance ratio — max the min of unclipped and clipped advantage terms — to stay trust-region-like without a hard KL constraint; the workhorse of RLHF and the baseline GRPO simplifies.

## The objective

Let rₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ) be the probability ratio between the updated and rollout policies, and Âₜ the (GAE) advantage estimate:

L^CLIP(θ) = E_t[ min( rₜ(θ)·Âₜ , clip(rₜ(θ), 1−ε, 1+ε)·Âₜ ) ]

- Maximize L^CLIP (so the sign convention favors raising advantaged actions).
- ε ≈ 0.1–0.3. Clipping caps the ratio so a single batch can't move the policy too far.
- The **min** makes it a pessimistic lower bound: the bonus from a good action (Â>0) is capped at 1+ε, while a bad action (Â<0) is penalized with no lower bound on how much you back off (clamped at 1−ε from above). The clip only "kicks in" when it would make the objective *better* — it never rewards moving further.
- Full PPO loss: L^CLIP − c₁·L^VF + c₂·entropy (value-function MSE + entropy bonus for exploration).
- Multiple SGD epochs over the same rollout batch; ratios drift from 1 across epochs, which is exactly what clipping bounds.

## Where it appears

- **InstructLM / RLHF** (Ouyang et al., 2022; Anthropic, Llama-2/3) — the PPO step that optimizes a reward model's score with a per-token KL penalty to the reference policy folded into the reward.
- **RLVR / reasoning RL** (o1-style, DeepSeek-R1) — PPO/GRPO maximize verifiable rewards on math/code; the clip + KL keep the policy from collapsing.
- **GRPO** (DeepSeek) — drops the value network, replaces Âₜ with group-normalized rewards over k sampled completions, but keeps the *same* clipped ratio objective.

## Common mistake

Thinking the clip is a hard trust region or a KL constraint — it is neither. Clipping a ratio per-sample gives no global guarantee on the policy shift, and because of the min, once a ratio is clipped its gradient is *zero*, not a penalty pulling it back. That is why practical PPO/RLHF still adds an explicit KL-to-reference term and early-stops on KL; clipping alone does not bound divergence.

## See also
- [[generalized-advantage-estimation]] — supplies the Âₜ plugged into the ratio
- [[importance-sampling-the-off-policy-ratio]] — rₜ(θ) is the IS ratio the clip tames
- [[grpo]] — same clipped objective, value-free group-baseline advantages
