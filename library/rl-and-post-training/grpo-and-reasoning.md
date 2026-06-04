# GRPO & Reasoning RL

*How DeepSeek-R1 made reasoning emerge from pure RL — and why it matters.*

---

## Why PPO is awkward for LLMs

PPO needs a **critic network** to estimate the value `V(s)` at each token. For language models that critic is expensive:

- It must be roughly as large as the policy to understand the state.
- It roughly doubles memory and compute.
- It is noisy and hard to fit when rewards are sparse (one signal at the end of a long sequence).

**GRPO (Group Relative Policy Optimization)** drops the critic entirely.

---

## GRPO — the idea

Instead of estimating a baseline from a critic, estimate it from a **group of sampled responses** to the same prompt.

For a prompt `q`:

1. Sample `G` responses `{o₁, …, o_G}` from the current policy.
2. Score each with reward `rᵢ` (e.g. 1 if correct, 0 if wrong).
3. Group baseline: `r̄ = mean(rᵢ)`.
4. Advantage: `Aᵢ = (rᵢ − r̄) / std(rᵢ)`.

The advantage answers one question: *was this response better or worse than the average response to this prompt?*

**Update rule:** PPO's clipped surrogate objective, but with these group-relative advantages in place of critic estimates.

**Why it works:** the group is its own baseline, so no second network is needed. Cheap and stable.

> One known wrinkle: the original length normalization can bias updates toward longer outputs. Follow-ups (e.g. **Dr. GRPO**, 2025) adjust the normalization to remove this. Most current libraries expose both variants.

---

## Verifiable rewards (RLVR)

GRPO shines when rewards are **verifiable**: a math answer is right or wrong; code passes tests or it doesn't. No reward model is needed — you just check the answer.

This is **RLVR (RL with Verifiable Rewards)**, formalized in AI2's Tülu 3.

| | RLHF preference model | RLVR verifiable reward |
|---|---|---|
| Source of signal | Learned from human labels | Rule / checker / test suite |
| Cost | Expensive to collect & train | Effectively free |
| Failure mode | Gameable, noisy | Hard to hack on closed tasks |
| Scope | Any subjective quality | Tasks with a checkable answer |

The trade-off: RLVR only applies where correctness is checkable, so it complements rather than replaces preference-based methods.

---

## DeepSeek-R1-Zero — reasoning from scratch

The headline experiment: take **DeepSeek-V3-Base** (next-token pretraining only, no SFT) and apply GRPO with:

- **Rule-based rewards only** — correctness (1/0) plus format following.
- **No human demonstrations.**
- **No SFT cold start.**

What happened:

- pass@1 on AIME 2024 rose from ~15.6% to ~71% (and higher with majority voting).
- **Emergent behavior:** the model spontaneously began showing its work, self-correcting, and reconsidering answers — none of which was in the reward signal.

The existence proof: **reasoning is RL-inducible from scratch.**

### Why those behaviors emerged

Self-reflection and verification *help predict correct answers*. A model that checks its work earns more reward, so optimization pressure alone discovers the strategy — no one told it to "think step by step." Complex cognitive behavior can fall out of a simple objective.

---

## DeepSeek-R1 — the practical version

R1-Zero works but is rough: mixed languages, poor readability, repetition. **DeepSeek-R1** cleans this up with a multi-stage pipeline:

1. **Cold-start SFT** — a small set of high-quality reasoning traces to initialize.
2. **Reasoning RL** — GRPO with verifiable rewards.
3. **Rejection-sampling SFT** — fine-tune on the best RL outputs.
4. **Final RL** — broaden to general helpfulness/harmlessness.
5. **Distillation** — compress R1's reasoning traces into smaller dense models.

The result matched o1-level reasoning on math and code benchmarks, with open weights. By 2025-2026 this recipe became the default template: nearly every open reasoning model (Qwen, Llama-derived distills, and others) follows the same SFT → verifiable-reward RL → distill arc.

---

## The inference-time scaling connection

Reasoning RL pairs naturally with **inference-time scaling** — spending more compute at decode time for better answers:

- **Best-of-N / majority vote:** sample N responses, pick the best or most common.
- **Tree search (MCTS) over reasoning steps:** search intermediate states.
- **Process reward models (PRMs):** reward each step, not just the final answer.

Snell et al. (2024) showed that for a fixed compute budget, a smaller model with more inference-time search can beat a larger model decoding greedily — though the win depends on having a good verifier or reward signal.

---

## Quick summary

| Method | What it needs | What it learns |
|---|---|---|
| RLHF + PPO | Human preferences + critic | Helpful, harmless behavior |
| DPO | Preference pairs (no RL loop) | Preferred over dispreferred outputs |
| GRPO + RLVR | Verifiable rewards only | Reasoning, problem-solving |

The standard 2025-2026 post-training stack: **SFT** (instruction behavior) → **DPO** (preference refinement) → **GRPO/RLVR** (reasoning), often with a distillation pass to ship smaller models.

See the [concept-library index](../bricks/README.md) for related notes.
