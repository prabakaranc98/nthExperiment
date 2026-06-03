# RL Fundamentals

*Reinforcement learning from scratch — no prior RL assumed.*

---

## The setup

A **Markov Decision Process (MDP)** is the formal framework for RL:

- **Agent** — the thing that makes decisions (your model)
- **Environment** — the world it acts in
- **State (s)** — a description of the current situation
- **Action (a)** — what the agent does
- **Reward (r)** — a signal telling the agent how good an action was
- **Policy (π)** — the agent's strategy: a function from states to actions (or distributions over actions)

The agent's goal: **maximize the total expected reward over time**.

---

## Return — the thing we're actually maximizing

A single reward is not the goal. The goal is the **return**: the total future reward from this point on.

**Discounted return:**
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

γ (gamma) is the discount factor (0 < γ ≤ 1). It weights immediate rewards more than distant ones.

Why discount? Mathematically: ensures the sum converges. Practically: near-term rewards are more certain.

---

## Value functions — expected future reward

**State value V(s):** expected return if you start in state s and follow policy π from there
**Action value Q(s, a):** expected return if you take action a in state s, then follow policy π

**The Bellman equation** (the fundamental recursion in RL):
```
V(s) = E[r + γ·V(s')]
```
The value of a state = immediate reward + discounted value of the next state.

Q-learning and value iteration both solve versions of this equation.

---

## Policy gradient — directly optimizing the policy

Instead of learning value functions and deriving a policy, directly optimize the policy.

**The policy gradient theorem:**
```
∇J(θ) = E[∇ log π_θ(a|s) · Q(s, a)]
```

In English: the gradient of expected return = expected value of (the gradient of log-probability of the action taken) × (how good that action was).

**REINFORCE** — the simplest algorithm:
1. Sample a trajectory (sequence of states, actions, rewards) using the current policy
2. Compute the return G_t at each timestep
3. Update: θ ← θ + η · Σ ∇ log π(aₜ|sₜ) · G_t

**Problem:** high variance. The return is a noisy estimate. Use a **baseline** (usually the value function) to reduce variance.

**Advantage A(s, a) = Q(s, a) - V(s)** — how much better is this action than average? Using advantage instead of raw return reduces variance dramatically.

---

## PPO — the practical default

Proximal Policy Optimization. Two core ideas:

1. **Trust region:** don't change the policy too much in one step. Limit the ratio of new/old policy probabilities:
   ```
   r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
   ```

2. **Clipped objective:** clip r_t to [1-ε, 1+ε]. This prevents large updates that destabilize training:
   ```
   L = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
   ```

PPO is the workhorse of RLHF. It's reliable, parallelizes well, and doesn't require a trust-region constraint (which would be expensive to compute).

---

## The exploration-exploitation tradeoff

You can't just exploit what you know — you need to explore to discover better strategies.

- **Exploit:** choose the action you currently think is best
- **Explore:** try something new that might be better

In language model RL, this is handled naturally by the model's sampling temperature and the stochastic nature of generation.

---

## RL in language models — what's different

In LLM RL:
- **State** = the prompt + generated tokens so far
- **Action** = next token to generate
- **Reward** = assigned *once* at the end of generation (sparse reward)
- **Policy** = the language model itself

The sparse reward problem (only getting a reward at the end of a long generation) is hard. Solutions: process reward models (reward each step), reward shaping, or GRPO (group-relative advantage).

---

## The key insight for 2025

RL doesn't just improve behavior — it can *create* new capabilities. DeepSeek-R1-Zero showed that a model trained with only rule-based rewards (right/wrong answer) develops reasoning behaviors like self-reflection and verification that it wasn't explicitly trained for. This is the existence proof that RL can be generative, not just refinement.
