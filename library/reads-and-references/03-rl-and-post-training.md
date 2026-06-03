# RL & Post-training

From RL fundamentals through RLHF, DPO, GRPO, and reasoning RL — the resources that make the 2024–2026 post-training revolution legible.

---

## RL foundations (Spinning Up)

OpenAI's Spinning Up is the best structured introduction to deep RL — combines theory with working implementations.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Spinning Up: Part 1 — Key Concepts in RL](https://spinningup.openai.com/en/spinningup/rl_intro.html) | MDPs, states, actions, rewards, policies, value functions — the complete vocabulary. | 🟢 |
| [Spinning Up: Part 2 — Kinds of RL Algorithms](https://spinningup.openai.com/en/spinningup/rl_intro2.html) | Taxonomy: model-free vs. model-based, policy gradient vs. Q-learning, on-policy vs. off-policy. The map of RL. | 🟢 |
| [Spinning Up: Part 3 — Intro to Policy Optimization](https://spinningup.openai.com/en/spinningup/rl_intro3.html) | Policy gradients, REINFORCE, advantage functions, GAE, PPO — the math behind the post-training stack. | 🟡 |
| [Spinning Up: Key Papers in RL](https://spinningup.openai.com/en/spinningup/keypapers.html) | Curated list of the 50 most important RL papers with brief descriptions. The RL reading map. | 🟢 |

---

## RL in practice (Karpathy)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) | Policy gradients + REINFORCE from scratch. The most intuitive RL implementation tutorial. Build this before reading PPO. | 🟡 |

---

## Post-training surveys (Lilian Weng)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | The taxonomy of reward hacking — specification gaming, Goodhart's law, proxy reward failures. Directly relevant to RLHF limitations and why RLVR matters. | 🟡 |
| [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | The full agent survey: planning, memory, tool use, ReAct, multi-agent systems. The best single-resource overview of the agentic frontier. | 🟡 |
| [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | Chain-of-thought, self-consistency, tree-of-thought, ReAct — all the prompting strategies that bridge to reasoning RL. | 🟡 |
| [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) | Jailbreaks, prompt injection, GCG — understanding the adversarial surface of LLMs post-training is supposed to defend against. | 🟡 |
| [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) | The thinking/reasoning model era — inference-time compute, chain-of-thought, process rewards, the o1/R1 paradigm. Current. | 🟡 |

---

## Thinking about data quality for post-training

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Thinking about High-Quality Human Data](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) | What makes SFT data good — quality vs. quantity, filtering, synthetic data. The data side of post-training. | 🟡 |

---

## Distill: RL visualization

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Understanding RL Vision](https://distill.pub/2020/understanding-rl-vision/) | How RL agents build visual representations — what they see and why. Interpretability meets RL. | 🟡 |
| [The Paths Perspective on Value Learning](https://distill.pub/2019/paths-perspective-on-value-learning/) | Value functions as sums over paths — a geometric interpretation that makes Bellman equations intuitive. | 🟡 |
