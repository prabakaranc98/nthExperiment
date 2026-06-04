# RL & Post-training

From RL fundamentals through RLHF, DPO, GRPO, and reasoning RL — the resources that make the 2024–2026 post-training revolution legible.

For the underlying concepts (policy gradients, KL control, reward models, verifiers), see the concept library: [`../bricks/README.md`](../bricks/README.md).

---

## Start here

If you read nothing else, read these three in order — fundamentals, the modern recipe, the reasoning-RL turn.

| Resource | What it is + why it's worth your time | Level |
|----------|----------------------------------------|-------|
| [Spinning Up: Part 1 — Key Concepts in RL](https://spinningup.openai.com/en/spinningup/rl_intro.html) | OpenAI's structured deep-RL intro. MDPs, states, actions, rewards, policies, value functions — the complete vocabulary in one sitting. | 🟢 |
| [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) | Karpathy builds policy gradients + REINFORCE from scratch. The most intuitive RL tutorial there is — do this before PPO. | 🟡 |
| [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) | Lilian Weng on the reasoning-model era: inference-time compute, chain-of-thought, process rewards, the o1/R1 paradigm. The best on-ramp to post-2024 RL. | 🟡 |

---

## Key papers

The canonical post-training stack. Where I'm confident of an arXiv ID I link it; otherwise the source is named so you can search it.

| Paper | What it is + why it matters | Year |
|-------|------------------------------|------|
| [InstructGPT — Training LMs to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | The RLHF recipe that defined modern post-training: SFT → reward model → PPO. Read it once, refer to it forever. | 2022 |
| [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | Skips the reward model and PPO loop — optimizes preferences directly with a simple classification loss. The default offline-alignment baseline. | 2023 |
| [Constitutional AI / RLAIF](https://arxiv.org/abs/2212.08073) | Replaces human preference labels with AI feedback against a written constitution. The basis for scalable, cheaper alignment. | 2022 |
| [DeepSeekMath — GRPO](https://arxiv.org/abs/2402.03300) | Introduces Group Relative Policy Optimization: drops the value model, normalizes advantages within a sampled group. The workhorse behind the reasoning-RL wave. | 2024 |
| [DeepSeek-R1 — incentivizing reasoning via RL](https://arxiv.org/abs/2501.12948) | RL-from-verifiable-rewards at scale (RLVR) producing emergent long-chain reasoning, with the R1-Zero pure-RL ablation. The defining open-weights reasoning paper. | 2025 |
| [Tülu 3 — pushing the frontier of open post-training](https://arxiv.org/abs/2411.15124) | AI2's fully open recipe (data, code, evals) spanning SFT, DPO, and RLVR. The most reproducible end-to-end post-training pipeline. | 2024 |
| [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) | Detailed, honest description of a frontier open model's SFT + multi-stage RL pipeline. Strong companion to the DeepSeek papers. | 2024 |
| Kimi k1.5 — scaling RL with LLMs | Long-context reasoning RL with length control and policy-optimization tricks; a useful contrast to the GRPO recipe. Search the Moonshot AI / Kimi technical report. | 2025 |

---

## Surveys & taxonomies

| Resource | What it is + why it's worth your time | Level |
|----------|----------------------------------------|-------|
| [Spinning Up: Key Papers in RL](https://spinningup.openai.com/en/spinningup/keypapers.html) | Curated map of ~50 foundational RL papers with one-line descriptions. The classical-RL reading list. | 🟢 |
| [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | Specification gaming, Goodhart's law, proxy-reward failure modes. Explains why RLHF breaks and why verifiable rewards matter. | 🟡 |
| [Thinking about High-Quality Human Data](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) | What makes SFT/preference data good — quality vs. quantity, filtering, synthetic data. The data side of post-training. | 🟡 |
| [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | Planning, memory, tool use, ReAct, multi-agent systems. Still the best single overview where post-training meets the agentic frontier. | 🟡 |

---

## Blogs & explainers

| Resource | What it is + why it's worth your time | Level |
|----------|----------------------------------------|-------|
| [Spinning Up: Part 2 — Kinds of RL Algorithms](https://spinningup.openai.com/en/spinningup/rl_intro2.html) | Model-free vs. model-based, policy-gradient vs. Q-learning, on- vs. off-policy. The map of the algorithm space. | 🟢 |
| [Spinning Up: Part 3 — Intro to Policy Optimization](https://spinningup.openai.com/en/spinningup/rl_intro3.html) | REINFORCE, advantage functions, GAE, PPO — the math under the entire post-training stack. | 🟡 |
| [The Paths Perspective on Value Learning](https://distill.pub/2019/paths-perspective-on-value-learning/) | Value functions as sums over paths — a geometric reading that makes Bellman equations click. Timeless. | 🟡 |
| [Understanding RL Vision](https://distill.pub/2020/understanding-rl-vision/) | How RL agents build visual representations. Interpretability meets RL; still the cleanest treatment. | 🟡 |
| [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | Chain-of-thought, self-consistency, tree-of-thought, ReAct — the prompting strategies that bridge to reasoning RL. | 🟡 |
| [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) | Jailbreaks, prompt injection, GCG — the adversarial surface that alignment is meant to defend. | 🟡 |
| Interconnects (Nathan Lambert) | The most reliable running commentary on RLHF, DPO, GRPO, and open post-training recipes. Subscribe and read the GRPO/RLVR explainers. | 🟡 |

---

## Courses & talks

| Resource | What it is + why it's worth your time | Level |
|----------|----------------------------------------|-------|
| [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) | Free, hands-on, regularly updated. Train agents end-to-end; the most accessible practical course. | 🟢 |
| [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/) | Rigorous classical-RL foundations with lectures and assignments online. Pairs well with Spinning Up. | 🟡 |
| Hugging Face LLM / Alignment material (RLHF & DPO) | Practical walkthroughs of the post-training pipeline using TRL. Search the Hugging Face docs and course pages for current RLHF/DPO units. | 🟡 |

---

## Tools

| Tool | What it is + why it's worth your time |
|------|----------------------------------------|
| [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) | Hugging Face's library for SFT, reward modeling, DPO, PPO, and GRPO. The default starting point for hands-on post-training. |
| [verl (Volcano Engine RL)](https://github.com/volcengine/verl) | Scalable RLHF/RLVR framework built for large-model GRPO/PPO training. Behind several open reasoning-model reproductions. |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + vLLM + DeepSpeed RLHF framework geared to high-throughput multi-node training. |
| [open-r1](https://github.com/huggingface/open-r1) | Hugging Face's open reproduction of the DeepSeek-R1 reasoning-RL pipeline — code, data recipes, and evals you can run. |
| [OpenAI Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | The maintained successor to OpenAI Gym — the standard environment API for classical-RL experimentation. |
