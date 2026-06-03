# Frontier AI Labs & Blogs

Where the frontier work is published, discussed, and explained — the primary sources and the best secondary commentary.

---

## Primary research sources

| Source | What they publish | How to follow |
|--------|------------------|---------------|
| [Anthropic Research](https://anthropic.com/research) | Alignment, interpretability, safety evals, system cards, Claude research | anthropic.com/research |
| [Transformer Circuits](https://transformer-circuits.pub) | The full mechanistic interpretability program | transformer-circuits.pub |
| [Distill.pub](https://distill.pub) | Interactive ML explanations (hiatus since 2021, but the archive is invaluable) | distill.pub |
| [DeepMind / Google Research Blog](https://deepmind.google/research/publications/) | AlphaFold, AlphaProof, Gemini technical work | deepmind.google |
| [OpenAI Research](https://openai.com/research) | o1/o3, GPT-4/5, RLHF, Spinning Up | openai.com/research |
| [Meta AI Research (FAIR)](https://ai.meta.com/research/) | Llama, PyTorch, FAISS, speech/vision research | ai.meta.com/research |
| [HuggingFace Blog](https://huggingface.co/blog) | Open ML: practical guides, new model releases, research summaries | huggingface.co/blog |

---

## The essential blogs

| Blog | Author | Best posts | Level |
|------|--------|-----------|-------|
| [colah.github.io](https://colah.github.io) | Chris Olah (Anthropic) | Backprop, LSTMs, Visual Information Theory, Topology of NNs | 🟢–🟡 |
| [karpathy.github.io](https://karpathy.github.io) | Andrej Karpathy | A Recipe for Training NNs, RNN Effectiveness, Pong from Pixels | 🟢–🟡 |
| [lilianweng.github.io](https://lilianweng.github.io) | Lilian Weng (OpenAI) | Every survey on every frontier topic — the most reliable secondary source | 🟡–🔴 |
| [jalammar.github.io](https://jalammar.github.io) | Jay Alammar | Illustrated Transformer, GPT-2, BERT, Word2Vec | 🟢 |

---

## Lilian Weng: the complete essential list

Weng's posts are the most consistently high-quality survey-level ML writing available. The full list most relevant to this curriculum:

| Post | Topic | Link |
|------|-------|------|
| Why We Think (2025) | Reasoning models, thinking budgets | [→](https://lilianweng.github.io/posts/2025-05-01-thinking/) |
| Reward Hacking in RL (2024) | RLHF failure modes, Goodhart's law | [→](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) |
| Extrinsic Hallucinations in LLMs (2024) | LLM failures, evaluation | [→](https://lilianweng.github.io/posts/2024-07-07-hallucination/) |
| LLM Powered Autonomous Agents (2023) | Agents, tools, memory, planning | [→](https://lilianweng.github.io/posts/2023-06-23-agent/) |
| Prompt Engineering (2023) | CoT, self-consistency, ReAct | [→](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) |
| The Transformer Family v2 (2023) | All architecture variants | [→](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) |
| Large Transformer Inference Optimization (2023) | KV cache, quantization, serving | [→](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) |
| What are Diffusion Models? (2021) | DDPM, score matching, SDEs | [→](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |
| How to Train Really Large Models (2021) | Distributed training | [→](https://lilianweng.github.io/posts/2021-09-25-train-large/) |
| Contrastive Representation Learning (2021) | SimCLR, CLIP, InfoNCE | [→](https://lilianweng.github.io/posts/2021-05-31-contrastive/) |
| Some Math behind NTK (2022) | Neural Tangent Kernel | [→](https://lilianweng.github.io/posts/2022-09-08-ntk/) |
| Generalized Visual Language Models (2022) | VLMs, multimodal | [→](https://lilianweng.github.io/posts/2022-06-09-vlm/) |
| Adversarial Attacks on LLMs (2023) | Jailbreaks, GCG | [→](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) |

---

## Karpathy: the essential posts

| Post | Why | Link |
|------|-----|------|
| A Recipe for Training Neural Networks | Debugging discipline | [→](https://karpathy.github.io/2019/04/25/recipe/) |
| The Unreasonable Effectiveness of RNNs | Historical + still illuminating | [→](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) |
| Deep RL: Pong from Pixels | RL from scratch | [→](https://karpathy.github.io/2016/05/31/rl/) |
| Deep Neural Nets: 33 Years Ago and Now | Historical perspective | [→](https://karpathy.github.io/2022/03/14/lecun1989/) |

---

## ArXiv feeds (how to stay current)

The frontier moves on arXiv. Key sections:
- `cs.LG` — machine learning (broad)
- `cs.AI` — artificial intelligence
- `cs.CL` — computation and language (NLP/LLMs)
- `cs.CV` — computer vision
- `stat.ML` — statistics + ML theory

**Best tools for tracking arXiv:**
- [Semantic Scholar](https://semanticscholar.org) — paper discovery, influence tracking
- [Papers With Code](https://paperswithcode.com) — papers + code + benchmarks
- [Connected Papers](https://connectedpapers.com) — visual citation graph
- [Elicit](https://elicit.com) — AI-assisted paper search

---

## HuggingFace blog: practical frontier

| Post | Why | Link |
|------|-----|------|
| KV Caching Explained | Inference memory | [→](https://huggingface.co/blog/not-lain/kv-caching) |
| Mastering Tensor Dimensions | Implementation clarity | [→](https://huggingface.co/blog/not-lain/tensor-dims) |
| Profiling in PyTorch | Performance debugging | [→](https://huggingface.co/blog/torch-profiler) |
| Unlocking Asynchronicity in Continuous Batching | Serving throughput | [→](https://huggingface.co/blog/continuous_async) |
