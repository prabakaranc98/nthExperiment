# RL Fundamentals

*Reinforcement learning from scratch — no prior RL assumed.*

*FAIRE context: a concept brick. See the [concept library index](../bricks/README.md).*

---

## The setup: Markov Decision Processes

A **Markov Decision Process (MDP)** is the formal framework for RL. It has six pieces:

| Symbol | Name | What it is |
|---|---|---|
| — | **Agent** | The thing that makes decisions (your model) |
| — | **Environment** | The world it acts in |
| **s** | **State** | A description of the current situation |
| **a** | **Action** | What the agent does |
| **r** | **Reward** | A scalar signal telling the agent how good an action was |
| **π** | **Policy** | The agent's strategy — a map from states to actions (or to distributions over actions) |

The agent's goal: **maximize total expected reward over time.**

---

## Return — what we actually maximize

A single reward is not the goal. The goal is the **return**: total future reward from this point on.

**Discounted return:**
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

γ (gamma) is the **discount factor** (0 < γ ≤ 1), weighting immediate rewards above distant ones. Why discount?

- **Mathematically:** ensures the infinite sum converges.
- **Practically:** near-term rewards are more certain.

---

## Value functions — expected future reward

- **State value V(s):** expected return starting in state s and following policy π.
- **Action value Q(s, a):** expected return taking action a in state s, then following π.

**The Bellman equation** is the fundamental recursion in RL:
```
V(s) = E[r + γ·V(s')]
```
The value of a state = immediate reward + discounted value of the next state. Q-learning and value iteration both solve versions of this equation.

---

## Policy gradient — directly optimizing the policy

Instead of learning value functions and deriving a policy, optimize the policy directly.

**Policy gradient theorem:**
```
∇J(θ) = E[∇ log π_θ(a|s) · Q(s, a)]
```

In words: the gradient of expected return equals the expected value of (gradient of the log-probability of the action taken) × (how good that action was).

**REINFORCE** — the simplest algorithm:
1. Sample a trajectory (states, actions, rewards) using the current policy.
2. Compute the return G_t at each timestep.
3. Update: `θ ← θ + η · Σ ∇ log π(aₜ|sₜ) · G_t`.

**The problem: high variance.** The return G_t is a noisy estimate. Two standard fixes:

- **Baseline:** subtract a state-dependent baseline (usually V(s)) — leaves the gradient unbiased but cuts variance.
- **Advantage:** `A(s, a) = Q(s, a) − V(s)` — how much better is this action than average from state s? Using advantage in place of raw return reduces variance dramatically.

---

## PPO — the practical default

**Proximal Policy Optimization.** Two core ideas:

1. **Trust region** — don't move the policy too far in one step. Track the ratio of new to old action probabilities:
   ```
   r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
   ```

2. **Clipped objective** — clip that ratio to [1−ε, 1+ε], preventing destabilizing updates:
   ```
   L = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
   ```

PPO is reliable, parallelizes well, and approximates a trust region without the expensive constraint that earlier methods (e.g. TRPO) computed explicitly. For LLM post-training it has been the long-running baseline, though critic-free variants (below) now dominate frontier reasoning runs.

---

## The exploration–exploitation tradeoff

You can't only exploit what you know — you must explore to discover better strategies.

- **Exploit:** choose the action you currently believe is best.
- **Explore:** try something new that might be better.

In LLM RL this is handled largely by **sampling temperature** and the stochasticity of generation: the same prompt yields varied rollouts, supplying exploration for free.

---

## RL in language models — what's different

| MDP piece | In an LLM |
|---|---|
| **State** | The prompt + tokens generated so far |
| **Action** | The next token |
| **Reward** | Usually assigned *once* at the end of generation (sparse) |
| **Policy** | The language model itself |

The **sparse-reward problem** — one reward after a long generation — makes credit assignment hard. Common responses:

- **Process reward models (PRMs):** score intermediate steps, not just the final answer.
- **Reward shaping:** add dense intermediate signals.
- **GRPO (Group Relative Policy Optimization):** sample a *group* of rollouts per prompt and compute each rollout's advantage relative to the group mean. This drops the separate value-function critic, which is what makes it cheap and now standard for reasoning RL.

---

## The key insight (2025–2026)

RL doesn't just refine behavior — it can *create* new capabilities. **DeepSeek-R1-Zero** showed that a model trained with only rule-based rewards (right/wrong on verifiable tasks) develops reasoning behaviors — self-reflection, verification, backtracking — that were never explicitly taught.

This established the **RLVR** paradigm (RL from Verifiable Rewards): when correctness can be checked automatically (math, code, formal proofs), simple outcome rewards plus a critic-free method like GRPO scale to strong reasoning. It is the existence proof that RL on LLMs can be generative, not merely refinement.

---

*Next: see the [concept library index](../bricks/README.md) for related bricks on reward modeling and post-training.*
