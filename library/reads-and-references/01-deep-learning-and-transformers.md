# Deep Learning & Transformers

The visual, practical, and mathematical canon for understanding how neural networks and transformers work. These are the resources you read *before* reading the original papers — they give you the intuition that makes the math legible.

> FAIRE context — part of the [concept library](../bricks/README.md). Start with the intuition builders, then drop into the from-scratch builds and the interpretability work.

Each entry is tagged 🟢 beginner-friendly · 🟡 intermediate · 🔴 advanced.

---

## Start here

The shortest path from "I've heard of transformers" to "I understand the forward pass."

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [3Blue1Brown — Neural Networks series](https://www.3blue1brown.com/topics/neural-networks) | Grant Sanderson's animated series, now extended through transformers. The single best visual on-ramp; watch before reading anything else. | 🟢 |
| [3Blue1Brown — Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt) | Chapter 5: the full transformer forward pass, embeddings to logits, made geometric. | 🟢 |
| [3Blue1Brown — Attention, step by step](https://www.3blue1brown.com/lessons/attention) | Chapter 6: queries, keys, values, and multi-head attention visualized from scratch. The clearest attention explainer that exists. | 🟢 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Jay Alammar's definitive static walkthrough of self-attention and the full architecture. Still the standard companion to "Attention Is All You Need." | 🟢 |
| [Karpathy — Deep Dive into LLMs like ChatGPT](https://x.com/karpathy/status/1887211193099825254) | 3.5-hour 2025 talk covering the full training stack (pretraining → SFT → RLHF), hallucination, and tool use for a general audience. Find it on YouTube under the same title. | 🟢 |

---

## Key papers & primers

The canon, plus the readable explainers that decode it.

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer) | "Attention Is All You Need" annotated line-by-line with working PyTorch. Read after a visual primer. | 🟡 |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | Decoder-only transformers: causal masking, LM head, generation. Bridges to modern LLMs. | 🟢 |
| [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) | Encoder-only transformers, masked LM, bidirectional context — the BERT paradigm. | 🟢 |
| [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) | Lilian Weng's organized tour of the architecture zoo: sparse/linear attention, SSMs, efficient variants. | 🟡 |
| [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) | 2025 survey of test-time compute and chain-of-thought reasoning — the System 1/System 2 framing behind reasoning models. | 🟡 |
| [Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) | 2024 deep dive into what hallucination is, where it comes from, and how it's measured and mitigated. | 🟡 |

---

## Blogs & explainers

The mechanics — backprop, geometry, optimization, attention's origins.

### The mathematical mechanics (Olah)

Chris Olah's posts are precise and visual — he's particularly good at making the math feel geometric.

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) | Backprop derived from first principles as graph traversal. The cleanest explanation of how gradients flow. Essential. | 🟢 |
| [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | Why networks work: they continuously deform data manifolds to separate classes. A geometric view. | 🟡 |
| [Deep Learning, NLP, and Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) | What embeddings are and why distributed representations matter. Pre-transformer but foundational. | 🟢 |
| [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | The LSTM cell dissected with diagrams — why gating works and why RNNs preceded transformers. | 🟢 |
| [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | Entropy, KL divergence, mutual information, all visualized — the foundations of ML loss functions. | 🟢 |

### Practical training (Karpathy)

Practitioner insights from someone who has trained frontier models.

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | The diagnostic approach to debugging training: start simple, complexify systematically. Required reading before any training run. | 🟢 |
| [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Character-level generation — the first "wow, this works" moment. Historical but still revelatory. | 🟢 |
| [Deep Neural Nets: 33 Years Ago and 33 Years from Now](https://karpathy.github.io/2022/03/14/lecun1989/) | Re-training LeCun's 1989 net with modern tools. Reveals how constant the foundations have stayed. | 🟢 |

### Origins & interactive deep dives (Distill)

Distill.pub published the highest-quality interactive ML explanations until 2021. Timeless classics.

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) | The original interactive attention explainer — the conceptual ancestor of the transformer. | 🟡 |
| [Why Momentum Really Works](https://distill.pub/2017/momentum/) | The real math behind momentum via eigenvalue analysis of convergence. Ties to optimization theory. | 🟡 |
| [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) | Interactive tour of t-SNE's hyperparameters and failure modes. Critical for reading any embedding plot. | 🟢 |
| [Visualizing Neural Machine Translation (Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) | How attention was invented — Bahdanau attention, seq2seq, the alignment problem. | 🟢 |

### Interpretability (Transformer Circuits)

How modern frontier-lab interpretability reverse-engineers these models. See also the [interpretability reading list](../bricks/README.md).

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Transformer Circuits Thread](https://transformer-circuits.pub/) | Anthropic's ongoing publication — induction heads, superposition, attention as circuits. The center of mechanistic interpretability. | 🔴 |
| [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | 2024 work extracting interpretable features from Claude 3 Sonnet with sparse autoencoders. The clearest demo of what "features" inside an LLM look like. | 🔴 |

---

## Courses & talks

| Resource | What it covers | Level |
|----------|---------------|-------|
| [Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Build micrograd → makemore → GPT from scratch in code. The definitive hands-on series; do it with a keyboard, not just a screen. | 🟡 |
| [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/) | 2025 course building an LLM end-to-end: tokenization, architecture, kernels, parallelism, scaling laws, evaluation. Lectures are public on YouTube. | 🔴 |
| [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/) | Top-down practical approach: build first, understand later. The fastest way to get working code. | 🟢 |
| [Stanford CS231n](https://cs231n.stanford.edu/) | CNNs for visual recognition — classification, detection, segmentation. The complete vision DL course. | 🟡 |
| [Stanford CS224n](https://web.stanford.edu/class/cs224n/) | NLP with deep learning, from word vectors through transformers and LLMs. The complete NLP course. | 🟡 |
| [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf) | Classical ML fundamentals from Andrew Ng — regression, SVMs, EM, Bayesian methods. The rigorous base layer. | 🟡 |

---

## Tools & build-it-yourself

The fastest way to understand a transformer is to build one.

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) | From-scratch reproduction of GPT-2 (124M), committed step by step so you can walk the git history. Pairs with the Zero to Hero GPT lecture. | 🟡 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | 2025 full-stack ChatGPT clone in ~8k lines: tokenizer → pretrain → SFT → RL → web UI, trainable for ~$100. The clearest end-to-end view of the whole pipeline. | 🔴 |
| [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) | Sebastian Raschka's 2024 book — design, pretrain, and fine-tune an LLM step by step. Code at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch). | 🟡 |
