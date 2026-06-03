# Process vs Outcome Reward Models (PRM / ORM)

**One-liner:** ORMs score only the final answer; PRMs score each intermediate reasoning step for denser credit assignment — the core supervision-design axis in reasoning RL and verifier-guided search.

## The definition

A reasoning trajectory is a prompt x with steps s₁…s_T and final answer a.

- **ORM** r_φ(x, s₁…s_T) → one scalar (or P(correct)). Trained on outcome labels y ∈ {0,1} from a checker/gold answer. Sparse: one signal per rollout.
- **PRM** r_φ(x, s₁…s_t) → a scalar *per step* (typically P(step is on a correct path)). Dense: T signals per rollout.

PRM step labels are expensive. Two ways to get them:
- **Human-annotated** — PRM800K (Lightman et al., *Let's Verify Step by Step*, 2023): ~800K step-level correctness labels.
- **Automatic / MC rollout** — Math-Shepherd, OmegaPRM (2024): label step s_t by sampling K completions from s_t and setting the soft label to the empirical fraction reaching a correct answer (a Monte-Carlo Q-value estimate). No human in the loop.

Aggregate PRM scores over steps for a full-trajectory score via **min** (any bad step kills it; best empirically in Lightman) or product/mean of step probabilities.

## Where it appears

- **Best-of-N / verifier-guided search** — rerank N sampled solutions by PRM/ORM score; PRM min-aggregation beats ORM and majority vote on MATH (Lightman 2023).
- **Step-level search** — beam search / MCTS-style rollouts use the PRM as a per-node value to prune branches (rStar-Math, AlphaProof-style search, 2024-25).
- **RL signal** — PRM provides dense per-step reward / process advantage; but DeepSeek-R1 (2025) deliberately dropped PRMs for RLVR, using a pure rule-based *outcome* reward (an ORM-style binary checker) + GRPO, citing reward hacking and PRM training cost.

## Common mistake

Believing a learned PRM is a "free" dense reward you can optimize hard against. PRMs are the *most* hackable verifiers — RL will exploit the step-scorer's blind spots (well-formatted-but-wrong steps), so PRMs work best for *inference-time reranking/search*, not as the unconstrained RL objective. The 2025 trend is verifiable outcome rewards (RLVR) precisely to avoid this.

## See also
- [[rlvr]] — uses a rule-based outcome (ORM) signal, the alternative to learned PRMs
- [[reward-hacking-over-optimization]] — why aggressively optimizing a learned PRM fails
- [[mcts-style-inference-search]] — PRM step values guide tree search at inference time
