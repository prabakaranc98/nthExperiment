# 01 · Language Foundation Models

*The reference domain. Every other field measures its recipe against what worked here first.*

## The adaptation

- **Token** — subword units (BPE / byte-level), the unit that made an open vocabulary tractable.
- **Objective** — next-token prediction: dense, self-supervised, infinitely available, no labels.
- **Inductive bias** — almost none. Self-attention lets scale and data, not architecture, carry the load.
- **Verification signal** — once weak (human preference, RLHF); now often *verifiable* (math/code answers checked automatically → RLVR), which unlocked reasoning.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---------------|-----|------|------|----------------|
| Attention Is All You Need | Google | 2017 | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | The substrate every domain now borrows |
| GPT-3 | OpenAI | 2020 | [arXiv 2005.14165](https://arxiv.org/abs/2005.14165) | Scale → in-context learning, no fine-tuning |
| Chinchilla | DeepMind | 2022 | [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) | Compute-optimal data:param ratio |
| Llama 3 Herd | Meta | 2024 | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) | The full open recipe, end to end |
| Tülu 3 | Ai2 | 2024 | [arXiv 2411.15124](https://arxiv.org/abs/2411.15124) | Open post-training; named and scaled RLVR |
| DeepSeek-R1 | DeepSeek | 2025 | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | Pure RL → emergent reasoning, open weights |
| Qwen3 | Alibaba | 2025 | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) | Unified thinking / non-thinking + budget control |
| Kimi K2 | Moonshot | 2025 | [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) | Open-weight agentic frontier (SWE-bench, tool use) |

## Where it stands (2025-2026)

- **Closed frontier (mid-2026):** Claude Opus 4.8, GPT-5.5, Gemini 3.1 Pro, Grok 4.3 — clustered at the top of the Artificial Analysis index, separated more by task profile (coding vs. reasoning vs. writing) than by a single score.
- **Reasoning is the axis of progress.** Post-DeepSeek-R1, gains come from RL with verifiable rewards and inference-time scaling (longer chains, tool calls), not bigger base models.
- **Open weights closed the gap.** DeepSeek, Qwen3, and Kimi K2 land within reach of closed labs and ship full recipes, shifting the moat to data, RL infrastructure, and agentic scaffolding.
- **Hybrid thinking and adaptive compute** (a "thinking budget" knob) are now table stakes — one model, dialed reasoning depth per query.
- **Teams run a model portfolio:** cheap tier for classification, workhorse for daily tasks, premium reserved for hard agentic workflows.

## Transferred vs. reinvented

**Transferred directly to other domains**
- Transformer block + self-attention as the default backbone.
- Self-supervised pretraining → fine-tune / adapt; scaling laws as a planning tool.
- Preference and RL post-training (RLHF / DPO / RLVR) as an alignment layer.

**Had to be reinvented for language itself**
- Discrete subword tokenization (no natural "word" unit).
- In-context learning as the primary interface — prompting replaced task-specific heads.
- Verifiable-reward RL pipelines, built only once reasoning became the bottleneck.

## Open problems

- **Reasoning vs. memorization:** does RLVR create new capability or surface base-model behavior? Generalization beyond verifiable domains stays unsettled.
- **Evaluation rot:** frontier scores saturate and leak into training data; clean, contamination-free benchmarks are scarce.
- **Long-horizon agency:** reliable multi-step tool use and error recovery still trail single-turn quality.
- **Cost of inference-time scaling:** more thinking buys accuracy at steep latency and compute price — when is it worth it?

## See also

- [02 · Vision](../02-vision/README.md) — the first domain to copy the recipe wholesale (ViT, MAE).
- [10 · Multimodal](../10-multimodal/README.md) — where language becomes the shared interface across modalities.
- [14 · Code](../14-code/README.md) — language's most verifiable subdomain; RLVR's proving ground.
- [Concept library](../../../library/bricks/README.md) — tokenization, scaling laws, attention as reusable bricks.
