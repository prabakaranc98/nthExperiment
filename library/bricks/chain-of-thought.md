# Chain-of-Thought & Test-Time Compute

**One-liner:** Elicit intermediate reasoning tokens before the answer, then spend more inference compute (longer chains, sampling, search, verification) to trade tokens for accuracy — the second scaling axis alongside pretraining.

## The mechanism

Standard prompting: model emits p(answer | question). CoT factors the answer through a latent reasoning trace z:

p(answer | question) = Σ_z p(answer | z, question) · p(z | question)

Sampling z and conditioning on it lets the model allocate more sequential compute (more forward passes / more tokens) to a hard problem, decomposing it into steps each of which is in-distribution. Triggered zero-shot by "Let's think step by step" (Kojima 2022) or few-shot exemplars with worked solutions (Wei 2022).

## Test-time compute knobs

- **Longer chains** — more reasoning tokens per sample (the o1/R1 regime; RL-trained to use them well).
- **Self-consistency** — sample k chains at temperature > 0, majority-vote the final answers (Wang 2022). Accuracy rises with k, then saturates.
- **Best-of-N / verifier reranking** — sample N, score each with a reward/verifier model, pick the best (PRM = process reward model scores each step; ORM = outcome only).
- **Search** — tree/beam search over reasoning steps (Tree-of-Thoughts), guided by a value function.

Test-time scaling law (Snell 2024; o1): accuracy is roughly log-linear in test-time compute, and on hard problems a small model + more inference compute can beat a larger model + greedy decoding.

## Where it appears

- **OpenAI o1 / o3, DeepSeek-R1, Gemini 2.x "thinking"** — RL (GRPO-style) trains long internal CoT against verifiable rewards (math/code); the chain is the product, not a prompt trick.
- **Self-consistency / GSM8K, MATH, AIME** — the standard way to squeeze accuracy from a fixed base model.
- **Agentic / tool-use loops** — interleave reasoning with actions (ReAct); each step is a CoT segment.

## Common mistake

Treating the printed chain as a faithful causal account of the computation. CoT improves *accuracy*, but the verbalized steps are often post-hoc and unfaithful — the model can reach the answer via different internal computation and rationalize it (Turpin 2023). Do not use the chain as an interpretability oracle or a safety guarantee.

## See also
- [[in-context-learning]] — few-shot CoT is ICL with reasoning exemplars
- [[grpo]] — the RL objective used to train long-chain reasoning models
- [[speculative-decoding]] — orthogonal way to cut the latency cost of long chains
