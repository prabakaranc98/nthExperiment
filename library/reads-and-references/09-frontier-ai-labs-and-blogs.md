# Frontier AI Labs & Blogs

Where the frontier work is published, discussed, and explained — the primary sources and the best secondary commentary. Curated for recency: 2024–2026 material is prioritized, dead links are dropped, and the timeless classics are kept. For the underlying concepts these resources cover, see the [concept library index](../bricks/README.md).

**Difficulty scale:** 🟢 Accessible · 🟡 Intermediate · 🔴 Advanced

---

## Start here

If you only follow a handful of sources, follow these. They have the highest signal and the most consistent throughput.

| Source | What it is | Why it's worth your time |
|--------|-----------|--------------------------|
| [Anthropic Research](https://anthropic.com/research) | Alignment, interpretability, safety evals, system cards | The deepest public work on what models actually do internally; system cards set the bar for honest capability + risk reporting. 🟡 |
| [Transformer Circuits](https://transformer-circuits.pub) | The full mechanistic interpretability program | One coherent, chronological research thread. The vocabulary the whole field now uses. 🔴 |
| [lilianweng.github.io](https://lilianweng.github.io) | Lilian Weng's survey-level essays | The single most reliable secondary source — each post is a self-contained literature review on a frontier topic. 🟡–🔴 |
| [Hugging Face Blog](https://huggingface.co/blog) | Practical open-ML guides + release notes | Where new open models, training recipes, and serving tricks land first, with runnable code. 🟢–🟡 |

---

## Primary research sources

The labs, publishing directly. Follow the research index, not the marketing blog.

| Source | What they publish | How to follow |
|--------|------------------|---------------|
| [Anthropic Research](https://anthropic.com/research) | Alignment, interpretability, safety evals, Claude system cards | anthropic.com/research |
| [Transformer Circuits](https://transformer-circuits.pub) | Mechanistic interpretability (SAEs, circuits, attribution graphs) | transformer-circuits.pub |
| [DeepMind / Google Research](https://deepmind.google/research/publications/) | Gemini, AlphaFold, AlphaProof/AlphaGeometry, agents | deepmind.google |
| [OpenAI Research](https://openai.com/research) | Reasoning models (o-series), GPT line, RLHF, Spinning Up | openai.com/research |
| [Meta AI Research (FAIR)](https://ai.meta.com/research/) | Llama, PyTorch, FAISS, speech/vision, world models | ai.meta.com/research |
| [Hugging Face Blog](https://huggingface.co/blog) | Open ML: practical guides, model releases, research recaps | huggingface.co/blog |
| [Distill.pub](https://distill.pub) | Interactive ML explanations | distill.pub — archive only (no longer publishing), but the back catalog remains some of the best ML writing ever produced |

> New since the last revision: open-weight reasoning models (e.g. the DeepSeek-R1 release and report) and their RL-from-verifiable-rewards recipes reshaped the field in 2025. The labs' own technical reports — read alongside Weng's "Why We Think" — are the primary sources here. Find them via the lab pages above and the arXiv tracking tools below.

---

## Key papers & technical reports

Frontier work increasingly ships as lab technical reports and arXiv preprints rather than venue papers. These are the canonical, durable references. Locate them via the tracking tools below — only verified links are inlined to avoid stale or wrong IDs.

| Resource | What it is | Why it matters |
|----------|-----------|----------------|
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | Anthropic, 2021 | The foundational interpretability paper — residual stream, attention as low-rank, virtual weights. 🔴 |
| [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) | Anthropic, 2022 | Why features superpose, and why SAEs are needed. 🔴 |
| [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | Anthropic, 2024 | SAEs scaled to a production model (Claude 3 Sonnet); interpretable features at scale. 🔴 |
| Attribution graphs / "On the Biology of a Large Language Model" | Anthropic, 2025 | Circuit-level tracing of real model behaviors via attribution graphs — the current frontier of mechanistic interp. Find on transformer-circuits.pub. 🔴 |
| DeepSeek-R1 technical report | DeepSeek, 2025 | The open recipe that popularized large-scale RL on verifiable rewards for reasoning. Find via arXiv / the DeepSeek release. 🟡 |

---

## Blogs & explainers

The essential individual writers. Classics that still teach well are kept; recency is added via Weng's 2024–2025 posts.

| Blog | Author | Best posts | Level |
|------|--------|-----------|-------|
| [lilianweng.github.io](https://lilianweng.github.io) | Lilian Weng | Survey-level essays on nearly every frontier topic (see full list below) | 🟡–🔴 |
| [colah.github.io](https://colah.github.io) | Chris Olah | Backprop, Understanding LSTMs, Visual Information Theory, Topology of NNs | 🟢–🟡 |
| [karpathy.github.io](https://karpathy.github.io) | Andrej Karpathy | A Recipe for Training NNs, RNN Effectiveness, Pong from Pixels | 🟢–🟡 |
| [jalammar.github.io](https://jalammar.github.io) | Jay Alammar | The Illustrated Transformer, GPT-2, BERT, Word2Vec | 🟢 |

### Lilian Weng — the complete essential list

Weng's posts are the most consistently high-quality survey-level ML writing available. Ordered newest-first so the recent frontier topics surface immediately.

| Post | Topic | Link |
|------|-------|------|
| Why We Think (2025) | Reasoning models, thinking budgets, test-time compute | [→](https://lilianweng.github.io/posts/2025-05-01-thinking/) |
| Reward Hacking in RL (2024) | RLHF failure modes, Goodhart's law | [→](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) |
| Extrinsic Hallucinations in LLMs (2024) | LLM failure modes, evaluation | [→](https://lilianweng.github.io/posts/2024-07-07-hallucination/) |
| LLM Powered Autonomous Agents (2023) | Agents: tools, memory, planning | [→](https://lilianweng.github.io/posts/2023-06-23-agent/) |
| Adversarial Attacks on LLMs (2023) | Jailbreaks, GCG | [→](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) |
| Prompt Engineering (2023) | CoT, self-consistency, ReAct | [→](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) |
| The Transformer Family v2 (2023) | All architecture variants | [→](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) |
| Large Transformer Inference Optimization (2023) | KV cache, quantization, serving | [→](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) |
| Generalized Visual Language Models (2022) | VLMs, multimodal | [→](https://lilianweng.github.io/posts/2022-06-09-vlm/) |
| Some Math behind NTK (2022) | Neural Tangent Kernel | [→](https://lilianweng.github.io/posts/2022-09-08-ntk/) |
| How to Train Really Large Models (2021) | Distributed training | [→](https://lilianweng.github.io/posts/2021-09-25-train-large/) |
| What are Diffusion Models? (2021) | DDPM, score matching, SDEs | [→](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |
| Contrastive Representation Learning (2021) | SimCLR, CLIP, InfoNCE | [→](https://lilianweng.github.io/posts/2021-05-31-contrastive/) |

### Karpathy — the essential posts

Older but timeless — the debugging discipline and from-scratch intuition still hold.

| Post | Why | Link |
|------|-----|------|
| A Recipe for Training Neural Networks | The debugging discipline every practitioner needs | [→](https://karpathy.github.io/2019/04/25/recipe/) |
| Deep Neural Nets: 33 Years Ago and Now | Historical perspective on what has (and hasn't) changed | [→](https://karpathy.github.io/2022/03/14/lecun1989/) |
| The Unreasonable Effectiveness of RNNs | Historical, still illuminating on sequence modeling | [→](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) |
| Deep RL: Pong from Pixels | Policy gradients from scratch | [→](https://karpathy.github.io/2016/05/31/rl/) |

### Hugging Face blog — the practical frontier

Concrete, runnable, and current. The fastest way to turn a paper into working code.

| Post | Why | Link |
|------|-----|------|
| Unlocking Asynchronicity in Continuous Batching | Serving throughput for LLM inference | [→](https://huggingface.co/blog/continuous_async) |
| KV Caching Explained | The core inference-memory mechanism | [→](https://huggingface.co/blog/not-lain/kv-caching) |
| Profiling in PyTorch | Performance debugging | [→](https://huggingface.co/blog/torch-profiler) |
| Mastering Tensor Dimensions | Implementation clarity | [→](https://huggingface.co/blog/not-lain/tensor-dims) |

---

## Courses & talks

For structured, video-first learning. Karpathy's recent work is the best current on-ramp to building models from scratch.

| Resource | What it is | Why it's worth your time |
|----------|-----------|--------------------------|
| Andrej Karpathy — Neural Networks: Zero to Hero | YouTube lecture series | Builds backprop → GPT from scratch, in code. The best hands-on intro available. Find on Karpathy's YouTube channel. 🟢–🟡 |
| Andrej Karpathy — "Let's build the GPT Tokenizer" / "Let's reproduce GPT-2 (nanoGPT)" | Long-form build videos (2024) | End-to-end reproduction of real models; demystifies tokenization and pretraining. 🟡 |
| [Spinning Up in Deep RL](https://spinningup.openai.com/) | OpenAI course + key-papers list | Still the best structured deep-RL on-ramp; the [Key Papers](https://spinningup.openai.com/en/spinningup/keypapers.html) list remains a durable reading map. 🟡 |
| 3Blue1Brown — Neural Networks series | YouTube visual course | The clearest visual intuition for backprop, attention, and transformers; the attention/transformer chapters are recent. Find on the 3Blue1Brown channel. 🟢 |

---

## Tools — staying current on arXiv

The frontier moves on arXiv first. Track these sections and use a discovery tool to filter the firehose.

**Key sections:** `cs.LG` (machine learning) · `cs.AI` (artificial intelligence) · `cs.CL` (NLP / LLMs) · `cs.CV` (vision) · `stat.ML` (theory)

| Tool | What it is | Why it's worth your time |
|------|-----------|--------------------------|
| [Semantic Scholar](https://semanticscholar.org) | Paper discovery + influence tracking | Citation-aware search and author feeds; good for tracing what builds on what. |
| [Connected Papers](https://connectedpapers.com) | Visual citation graph | Drop in one paper, see its neighborhood — fastest way to map a subfield. |
| [Papers With Code](https://paperswithcode.com) | Papers + code + benchmarks | Links results to implementations and leaderboards (note: actively maintained but verify current benchmark relevance). |
| [Elicit](https://elicit.com) | AI-assisted literature search | Summarizes and extracts findings across many papers at once. |
| [Hugging Face Papers](https://huggingface.co/papers) | Daily curated arXiv feed | Community-voted daily picks with discussion — a low-effort way to catch the day's notable releases. |
| [alphaXiv](https://www.alphaxiv.org/) | Open discussion layer over arXiv | Inline comments and trending papers; useful for seeing what researchers are actually debating. |
