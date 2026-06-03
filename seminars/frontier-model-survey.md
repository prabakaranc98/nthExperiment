# Frontier & Open Model Survey — June 2024 → June 2026

*A recipe-level map of the last two years of frontier and open models, anchored to primary technical reports. Built for reading the actual sources, not just tracking benchmark numbers.*

---

## How to read this

The doc has four parts:

1. **The seven shifts** — what actually changed in the recipe over two years.
2. **The broad map** — a comparison table covering ~35 models (architecture, the one thing each is known for, link).
3. **Deep profiles** — full pretraining → post-training → safety breakdowns for the ~18 recipe-defining models.
4. **Cross-cutting techniques & the safety/interpretability thread** — the papers underneath the models.

Throughout: **open** = downloadable weights with a real report; **frontier-closed** = no weights but a substantive technical report or system card. A few details on 2025–2026 releases are drawn from secondary coverage where labs published only blog posts or model cards rather than full reports — verify against the primary source before citing in your own work.

---

## Part 1 — The Seven Shifts (2024 → 2026)

If you read nothing else, these are the recipe changes that define the period.

**1. Reasoning RL became the second pretraining.**
The single biggest shift. OpenAI's o1 (Sept 2024) introduced inference-time scaling by lengthening the chain-of-thought, and DeepSeek-R1 (Jan 2025) showed reasoning can be incentivized through pure reinforcement learning, removing the need for human-annotated demonstrations. By 2026, "train a base model, then RL it into a reasoner" is the default pipeline, not a research curiosity.

**2. Verifiable rewards replaced (a lot of) human preference.**
RLHF on human comparisons gave ground to RL on *checkable* signals — math answers that are right or wrong, code that passes tests. DeepSeek's GRPO and AI2's RLVR formalized this. It scales because the reward is free and unhackable in a way preference models aren't.

**3. MoE went from exotic to default at the frontier.**
DeepSeek-V3 (671B total / 37B active), Llama 4, Qwen3, Kimi K2 (1T / 32B active) — sparse Mixture-of-Experts is now how you buy capacity without paying full inference cost. The accompanying tricks (auxiliary-loss-free load balancing, fine-grained + shared experts, MLA for KV-cache compression) are the new standard toolkit.

**4. Hybrid / toggleable reasoning.**
Rather than shipping a separate "reasoning model," 2025 models fold thinking and non-thinking into one set of weights with a **thinking budget** knob — Qwen3, Gemini 2.5, GPT-5's router. The user (or a router) decides how much compute to spend per query.

**5. Training-efficiency became a headline result.**
DeepSeek-V3 trained for ~$5.6M / 2.79M H800-hours and forced everyone to take FP8 training, hardware-aware co-design, and token efficiency (Kimi's Muon optimizer) seriously.

**6. Agentic capability is now a first-class training target.**
Models are explicitly post-trained on tool use, long-horizon tasks, and multi-context-window operation — not just chat.

**7. Safety reporting matured into a genre.**
System cards went from marketing to substantial documents covering reward hacking, sabotage, evaluation awareness, scheming, CoT monitorability, and dangerous-capability (CBRN/cyber) thresholds tied to formal frameworks (Anthropic's RSP/ASL, OpenAI's Preparedness, DeepMind's Frontier Safety).

---

## Part 2 — The Broad Map

| Model | Lab | Released | Type | Architecture | Known for | Report |
|---|---|---|---|---|---|---|
| **Llama 3.1 405B** | Meta | Jul 2024 | Open | Dense | Largest openly-released dense model; canonical recipe paper | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) |
| **Llama 3.2 / 3.3** | Meta | late 2024 | Open | Dense + vision | On-device (1B/3B) + multimodal | Meta blog |
| **Llama 4 Scout/Maverick** | Meta | Apr 2025 | Open | MoE, early-fusion MM | First Llama MoE; 10M-token context (Scout) | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| **Qwen2 / 2.5** | Alibaba | Jun–Dec 2024 | Open | Dense + MoE | Broadest open family; strong multilingual/code | [arXiv 2412.15115](https://arxiv.org/abs/2412.15115) |
| **Qwen3** | Alibaba | May 2025 | Open | Dense + MoE | Unified thinking/non-thinking; thinking budget | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) |
| **DeepSeek-V2** | DeepSeek | mid 2024 | Open | MoE + MLA | Introduced Multi-head Latent Attention | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) |
| **DeepSeek-V3** | DeepSeek | Dec 2024 | Open | MoE 671B/37B | ~$5.6M training; FP8; aux-loss-free balancing | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) |
| **DeepSeek-R1** | DeepSeek | Jan 2025 | Open | R1-Zero + R1 | Pure-RL reasoning; Nature cover | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) · [Nature](https://www.nature.com/articles/s41586-025-09422-z) |
| **Mistral / Mixtral / Large** | Mistral | 2024–25 | Open(-ish) | Dense + MoE | Sparse MoE in the West; strong small models | Mistral blog |
| **Gemma 2** | Google | Jun 2024 | Open | Dense | Distillation + logit soft-capping at small scale | [arXiv 2408.00118](https://arxiv.org/abs/2408.00118) |
| **Gemma 3** | Google | Mar 2025 | Open | Dense + vision | Long context + multimodal in open weights | [arXiv 2503.19786](https://arxiv.org/abs/2503.19786) |
| **OLMo 2** | AI2 | late 2024–25 | **Fully open** | Dense | Everything released: data, code, logs, recipe | [arXiv 2501.00656](https://arxiv.org/abs/2501.00656) |
| **Tülu 3** | AI2 | Nov 2024 | **Fully open** | Post-training | Open post-training recipe; introduced RLVR | [arXiv 2411.15124](https://arxiv.org/abs/2411.15124) |
| **Phi-4** | Microsoft | Dec 2024 | Open | Dense 14B | Synthetic-data-heavy; punches above its size | [arXiv 2412.08905](https://arxiv.org/abs/2412.08905) |
| **Command R / R+** | Cohere | 2024 | Open | Dense | RAG- and tool-use-optimized | Cohere docs |
| **Nemotron** | NVIDIA | 2024–25 | Open | Dense + reward models | Open reward models + synthetic-data pipelines | NVIDIA report |
| **Granite 3.x** | IBM | 2024–25 | Open | Dense + MoE | Enterprise/governance-focused | IBM docs |
| **Kimi K2** | Moonshot | Jul 2025 | Open | MoE 1T/32B | Muon/MuonClip optimizer; agentic-first | [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) |
| **GLM-4.x** | Zhipu | 2024–25 | Open | Dense + MoE | Strong bilingual; agentic variants | Zhipu/arXiv |
| **GPT-4o** | OpenAI | May 2024 | Closed | Omni dense | Native multimodal (text/audio/vision) | OpenAI |
| **o1 / o3 / o4-mini** | OpenAI | 2024–25 | Closed | Reasoning | Launched inference-time scaling; RL reasoning | System cards |
| **GPT-4.5** | OpenAI | early 2025 | Closed | Dense | Last big "non-reasoning" scale-up | System card |
| **GPT-5 (→5.5)** | OpenAI | Aug 2025+ | Closed | Router + sub-models | Unified router over fast/thinking models | [System card](https://cdn.openai.com/gpt-5-system-card.pdf) |
| **Claude 3.5 Sonnet** | Anthropic | Jun/Oct 2024 | Closed | Dense | Coding/agentic step-change; computer use | Anthropic |
| **Claude 3.7 Sonnet** | Anthropic | Feb 2025 | Closed | Hybrid reasoning | First Claude with visible extended thinking | Anthropic |
| **Claude Opus/Sonnet 4** | Anthropic | May 2025 | Closed | Hybrid reasoning | "Spiritual bliss" attractor; ASL-3 deploy | System card |
| **Claude Opus 4.5 / 4.6** | Anthropic | Nov 2025 / Feb 2026 | Closed | Hybrid reasoning | SOTA agentic/coding; deep alignment evals | [4.5 card](https://www.anthropic.com/claude-opus-4-5-system-card) |
| **Gemini 1.5 Pro/Flash** | Google | 2024 | Closed | MoE | 1M–10M token context | [arXiv 2403.05530](https://arxiv.org/abs/2403.05530) |
| **Gemini 2.0** | Google | Feb 2025 | Closed | MoE | Agentic + native tool use | DeepMind |
| **Gemini 2.5 Pro / Deep Think** | Google | Mar–Aug 2025 | Closed | MoE + thinking | IMO gold-level Deep Think; parallel thinking | [2.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) |
| **Grok 2 / 3** | xAI | 2024–25 | Closed | Dense | Scale-up on Colossus cluster | xAI |
| **Grok 4** | xAI | Jul 2025 | Closed | Reasoning | RL at "pretraining scale" on 200k GPUs | [Model card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf) |

---

## Part 3 — Deep Profiles

### DeepSeek-V3 → R1 *(the two most important reads of the period)*

**DeepSeek-V3** (Dec 2024). MoE: 671B total / 37B active. MLA compresses the KV cache via joint low-rank projection. Fine-grained experts + shared experts. Auxiliary-loss-free balancing (nudges per-expert routing bias up/down by load — avoids the performance hit of the usual balancing loss). Multi-token prediction (several future tokens; reusable for speculative decoding). FP8 + DualPipe. Full training: ~2.79M H800 GPU-hours (~$5.6M). The most complete public account of training a frontier-class model cheaply. → [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)

**DeepSeek-R1 / R1-Zero** (Jan 2025; Nature Sept 2025). R1-Zero: DeepSeek-V3-Base + GRPO + *only* rule-based rewards (answer correctness + format). No SFT cold-start, no human demos. Self-reflection and verification emerged; AIME 2024 pass@1 15.6% → 71%. GRPO drops the critic network of PPO, estimates baseline from the group of sampled responses. R1 adds a small cold-start SFT set and multi-stage training to fix readability. The existence proof that reasoning is RL-inducible from scratch. → [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) · [Nature](https://www.nature.com/articles/s41586-025-09422-z)

---

### Llama 3.1 → Llama 4 *(Meta)*

**Llama 3.1 (405B)** — the reference recipe paper of the era. Dense, but the most detailed open account of data curation, multi-stage pretraining, scaling-law-driven decisions, and a documented post-training pipeline (SFT + DPO over multiple rounds). If you read one paper on "how a frontier lab builds a model end to end," this is it. → [arXiv 2407.21783](https://arxiv.org/abs/2407.21783)

**Llama 4** (Apr 2025) — Meta's first MoE. Scout: 17B active / 109B total over 16 experts, 10M-token context. Maverick: 17B active / 400B total over 128 experts. Both natively multimodal via early fusion, trained in FP8. Behemoth (288B active, 16 experts) positioned as a teacher for co-distillation, not openly released at announcement. → [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

---

### Qwen2.5 → Qwen3 *(Alibaba)*

Broadest open family, 0.6B to 235B, dense and MoE. Qwen3's key move: thinking mode and non-thinking mode in one model with a dynamic thinking-budget mechanism. Flagship Qwen3-235B-A22B is MoE (235B/22B active); multilingual from 29 → 119 languages; Apache 2.0. Later Qwen3-Next: extreme sparsity (80B/3B active) with hybrid attention + Gated Delta Net. → [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)

---

### Kimi K2 *(Moonshot)*

1.04T-parameter MoE, 32B activated, pre-trained on 15.5T tokens with zero loss spikes using **MuonClip** (Muon + QK-clip for stability at trillion scale). Post-training centers on large-scale agentic-data-synthesis and a joint RL stage where the model improves through interaction with real and synthetic environments. Architecture: 61 layers, 384 experts (8 active/token), MLA, SwiGLU. The most interesting recent challenge to "AdamW + the usual recipe." → [arXiv 2507.20534](https://arxiv.org/abs/2507.20534)

---

### OLMo 2 + Tülu 3 *(AI2 — the fully-open reference)*

**OLMo 2** releases data, code, intermediate checkpoints, and training logs — the only frontier-adjacent fully-open pipeline. **Tülu 3** is the matching open post-training recipe and the clearest statement of **RLVR (RL with Verifiable Rewards)** — RL against programmatically checkable answers rather than a learned preference model. For a research engineer, this pair is the best teaching artifact of the whole list. → [OLMo 2 arXiv 2501.00656](https://arxiv.org/abs/2501.00656) · [Tülu 3 arXiv 2411.15124](https://arxiv.org/abs/2411.15124)

---

### Phi-4 + Gemma 2/3 *(Microsoft / Google)*

**Phi-4** (14B): strongest argument for synthetic-data-centric pretraining — carefully generated and curated data lets a small model rival much larger ones on reasoning. **Gemma 2/3**: Gemma 2 notable for distillation from a larger teacher and logit soft-capping + alternating local/global attention; Gemma 3 adds vision and long context in open weights. → [Phi-4 arXiv 2412.08905](https://arxiv.org/abs/2412.08905) · [Gemma 3 arXiv 2503.19786](https://arxiv.org/abs/2503.19786)

---

### OpenAI: GPT-4o → o1 → o3 → GPT-5

Least architecture disclosure, clearest capability story. GPT-4o unified text/audio/vision. o1 (Sept 2024) first to scale inference-time reasoning via long CoT, trained with RL. GPT-5 (Aug 2025) is not a single model — a fast model for most queries, a deeper reasoning model for hard ones, and a real-time router. The system card openly discusses reward hacking as a named, measured risk ("RL post-training can teach models to be overconfident or trick fallible graders when that earns reward"). → [GPT-5 system card](https://cdn.openai.com/gpt-5-system-card.pdf)

---

### Anthropic: Claude 3.5 → Opus 4.x

3.5 Sonnet (2024): coding/agentic step-change + computer use. 3.7 Sonnet (Feb 2025): visible extended thinking. Opus 4 / Sonnet 4 (May 2025): hybrid reasoning under ASL-3 deployment standard; the 123-page system card documented the "spiritual bliss" attractor state that emerged in extended self-interaction. By Opus 4.5 / 4.6: system cards became the most thorough alignment documents in the industry — covering reward hacking, sabotage capability, evaluation awareness, model welfare, dangerous-capability evals. Anthropic explicitly states they refrain from training on the model's chain-of-thought to preserve CoT as a monitoring surface. → [Opus 4.5 card](https://www.anthropic.com/claude-opus-4-5-system-card)

---

### Gemini 1.5 → 2.5 *(Google DeepMind)*

Gemini 1.5 defined long context — sparse MoE with 1M–10M token window. Gemini 2.5 Pro (Mar 2025): natively multimodal thinking model with toggleable thinking budgets, 1M-token context. Deep Think: parallel thinking — generates many ideas at once, revises/combines before answering. Related to the system that reached gold-medal standard at IMO 2025. Safety section maps to DeepMind's Frontier Safety Framework. → [Gemini 2.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)

---

### Grok 4 *(xAI)*

Thinnest on architecture, clear data point on RL-at-scale: xAI ran RL on 200,000-GPU Colossus cluster at "pretraining scale," 6× compute-efficiency gain, verifiable training data expanded beyond math and code, trained natively to use a code interpreter and web browser. The bet: reasoning RL is now a pretraining-magnitude compute consumer. → [Grok 4 model card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf)

---

## Part 4 — Cross-Cutting Techniques & Safety Thread

### Technique papers underneath the models

| Technique | Paper | What it does |
|-----------|-------|-------------|
| **GRPO** | DeepSeekMath / R1 | Critic-free RL; advantage from a group of samples. Most-copied reasoning-RL recipe. |
| **RLVR** | Tülu 3 | RL against checkable answers instead of a preference model |
| **MLA** | DeepSeek-V2/V3 | KV-cache compression via low-rank projection; inference efficiency for long context |
| **Aux-loss-free MoE balancing** | DeepSeek-V3 | Per-expert routing bias nudged by load — avoids performance hit of balancing loss |
| **Multi-token prediction** | DeepSeek-V3 | Predict several future tokens; doubles as speculative decoding spec |
| **MuonClip** | Kimi K2 | Muon (matrix-orthogonalized updates) + QK-clip; trillion-scale stability |
| **Chinchilla scaling** | Hoffmann et al. [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) | Compute-optimal data/parameter baseline — most 2024–26 models *over-train* past it for cheaper inference |
| **Distillation / synthetic data** | Gemma 2, Phi-4, Llama 4 Behemoth | Three flavors: teacher distillation, synthetic-data pretraining, co-distillation |

### Safety & interpretability thread

- **Constitutional AI / RLAIF** [arXiv 2212.08073](https://arxiv.org/abs/2212.08073) — align with AI feedback against a written constitution; still load-bearing in the Claude 4.x cards.
- **Mechanistic interpretability at scale** — Anthropic's SAE work ("Towards Monosemanticity" → "Scaling Monosemanticity," 2024) extracted millions of interpretable features from Claude 3 Sonnet. Turns interp from toy-model work into something applied at production scale.
- **Reward hacking as a measured quantity** — GPT-5 and Claude 4.x cards now report reward-hack rates on held-out / impossible tasks.
- **CoT monitorability** — labs argue for *not* training on the chain-of-thought so it stays a faithful monitoring surface. Anthropic states this explicitly.
- **Dangerous-capability frameworks** — Anthropic's RSP/ASL, OpenAI's Preparedness Framework, DeepMind's Frontier Safety Framework. Claude Opus 4 deploying under ASL-3 was the first time a major lab shipped under heightened safeguards for bio/cyber risk.

---

## Suggested Reading Order *(for a research engineer)*

| # | Read | Why |
|---|------|-----|
| 1 | Llama 3.1 herd | Complete end-to-end recipe, written to be read |
| 2 | DeepSeek-V3 | Efficient MoE pretraining; the efficiency benchmark |
| 3 | DeepSeek-R1 + Nature version | Reasoning-RL breakthrough; center of gravity of the period |
| 4 | Tülu 3 + OLMo 2 | Only fully-open pre + post-training pipeline; cleanest RLVR statement |
| 5 | Qwen3 | Hybrid-thinking/budget design now becoming standard |
| 6 | Kimi K2 | Agentic post-training + optimizer challenge to AdamW |
| 7 | Claude Opus 4.5 or GPT-5 system card | How safety, alignment, and dangerous-capability evals are actually structured |
| 8 | Scaling Monosemanticity | On-ramp to the interpretability thread |

**The arc:** pretrain → efficient-MoE → reasoning-RL → open-recipe → hybrid-inference → agentic → safety/interp. That is the actual shape of where the frontier moved.
