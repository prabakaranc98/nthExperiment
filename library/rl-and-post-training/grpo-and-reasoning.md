# GRPO & Reasoning RL

*How DeepSeek-R1 made reasoning emerge from pure RL — and why it matters.*

---

## The problem with PPO for LLMs

PPO needs a **critic network** to estimate value V(s) at each step. For language models:
- The critic must be as large as the model (to understand the state)
- This doubles memory and compute
- It's noisy and hard to train for sparse rewards

**GRPO (Group Relative Policy Optimization)** drops the critic entirely.

---

## GRPO — the idea

Instead of estimating baseline from a critic, estimate it from a **group of sampled responses**.

For a given prompt q:
1. Sample G responses {o₁, o₂, ..., o_G} from the current policy
2. Compute reward rᵢ for each response (e.g., 1 if correct, 0 if wrong)
3. Compute the **group baseline**: r̄ = mean(rᵢ)
4. **Advantage** for response i: Aᵢ = (rᵢ - r̄) / std(rᵢ)

The advantage is: "was this response better or worse than the average response to this prompt?"

**Update rule:** same as PPO's clipped objective, but using group-relative advantages instead of critic estimates.

**Why this works:** no extra network needed. The group itself provides the baseline. Cheap and stable.

---

## Verifiable rewards — why this matters

GRPO works best when rewards are **verifiable**: a math answer is right or wrong, code passes tests or doesn't. No reward model needed — just check the answer.

This is **RLVR (RL with Verifiable Rewards)**, formalized by AI2's Tülu 3.

**The advantage over RLHF:** human preference models can be gamed, have noise, and are expensive to train. A correct/incorrect signal is free, scalable, and unhackable.

---

## DeepSeek-R1-Zero — reasoning from scratch

The key experiment: take DeepSeek-V3-Base (an LLM trained only on next-token prediction, no SFT) and apply GRPO with:
- **Only rule-based rewards**: correctness (1/0) + format following
- **No human demonstrations**
- **No SFT cold start**

What happened:
- The model's pass@1 on AIME 2024 went from ~15.6% to ~71%
- **Emergent behavior**: the model spontaneously started showing its work, self-correcting, and reconsidering answers — none of which was in the training signal

This is the existence proof: **reasoning is RL-inducible from scratch**.

---

## Why reasoning behaviors emerged

The hypothesis: self-reflection and verification *help predict correct answers*. A model that checks its work gets more reward. RL finds this strategy because it increases return. No one told it to think step by step.

This is a deep result: complex cognitive behaviors can emerge from simple optimization pressure.

---

## DeepSeek-R1 — the practical version

R1-Zero works but has problems: mixed languages, poor readability, repetition. DeepSeek-R1 fixes this with:

1. **Cold-start SFT**: a small set of high-quality reasoning examples to initialize
2. **Multi-stage training**: SFT → RL → SFT (on best outputs) → RL again
3. **Distillation**: take R1's reasoning traces and distill into smaller, denser models

The result: R1 matches o1-level reasoning on math/code benchmarks. Fully open source.

---

## The inference-time scaling connection

GRPO and reasoning RL enable **inference-time scaling**: spend more compute at inference to get better answers.

Methods:
- **Best-of-N**: sample N responses, pick the best
- **MCTS over reasoning steps**: tree search over intermediate reasoning states
- **Process reward models (PRMs)**: reward each reasoning step, not just the final answer

The key insight from Snell et al. (2024): for a fixed compute budget, you can often do better by using a smaller model with more inference-time search than a larger model with greedy decoding.

---

## Quick summary

| Method | What it needs | What it learns |
|--------|--------------|---------------|
| RLHF + PPO | Human preferences + critic | Helpful, harmless behavior |
| DPO | Preference pairs (no RL) | Preferred outputs over dispreferred |
| GRPO + RLVR | Verifiable rewards only | Reasoning, problem-solving |

The 2025 recipe: SFT (instruct behavior) → DPO (refine preferences) → GRPO/RLVR (reasoning).
