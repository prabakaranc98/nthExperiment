# Deep Learning & Transformers

The visual, practical, and mathematical canon for understanding how neural networks and transformers work. These are the resources you read *before* reading the original papers — they give you the intuition that makes the math legible.

---

## Visual intuition first (Jalammar)

Jay Alammar's illustrated guides are the best entry point to transformers. They build mechanistic understanding through animation without hiding the math.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | The definitive visual walkthrough of self-attention and the full transformer architecture. Read before any transformer paper. | 🟢 |
| [Visualizing Neural Machine Translation / Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) | How attention was invented — Bahdanau attention, seq2seq, the alignment problem. The historical foundation. | 🟢 |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | Decoder-only transformer in detail: causal masking, language modeling head, how generation works. Bridges to modern LLMs. | 🟢 |
| [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) | Encoder-only transformers, masked LM, bidirectional context. The BERT paradigm. | 🟢 |
| [How GPT3 Works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) | Scale + few-shot learning. How GPT-3's size enables emergent behaviors. | 🟢 |
| [The Illustrated Retrieval Transformer (RETRO)](https://jalammar.github.io/illustrated-retrieval-transformer/) | Retrieval-augmented models — parametric + non-parametric memory. | 🟡 |

---

## The mathematical mechanics (Colah)

Chris Olah's posts are precise and visual — he's particularly good at making the math feel geometric.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) | Backprop derived from first principles as graph traversal. The cleanest explanation of how gradients flow. Essential. | 🟢 |
| [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | Why neural networks work: they learn to continuously deform data manifolds to separate classes. A geometric view. | 🟡 |
| [Deep Learning, NLP, and Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) | What word embeddings are, why distributed representations matter, the geometry of semantic space. Pre-transformer but foundational. | 🟢 |
| [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | The LSTM cell dissected with diagrams. Essential for understanding why gating works and why RNNs preceded transformers. | 🟢 |
| [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | Entropy, KL divergence, mutual information — all visualized. The information-theoretic foundations of ML loss functions. | 🟢 |
| [Conv Nets: A Modular Perspective](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) | CNNs as modular compositions — the right mental model for why depth works. | 🟢 |

---

## Practical training (Karpathy)

Andrej Karpathy writes from the trenches — these are practitioner insights from someone who has trained frontier models.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | The diagnostic approach to debugging training: start simple, complexify systematically, fix one thing at a time. Required reading before any training run. | 🟢 |
| [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Generating text character-by-character. The first "wow, this works" moment for many people. Historical but still revelatory. | 🟢 |
| [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) | Policy gradients + REINFORCE implemented from scratch, explained intuitively. The best first RL tutorial. | 🟡 |
| [Deep Neural Nets: 33 Years Ago and 33 Years from Now](https://karpathy.github.io/2022/03/14/lecun1989/) | Karpathy re-trains LeCun's 1989 network with modern tools. Reveals how much the foundations have stayed constant. Perspective-giving. | 🟢 |

---

## Survey-level understanding (Lilian Weng)

Lilian Weng's posts are the best comprehensive surveys — not papers, not tutorials, but structured synthesis across a topic.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) | Every major transformer variant: sparse attention, linear attention, SSMs, efficient attention. The architecture zoo, organized. | 🟡 |
| [Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/) | How vision and language are connected in models — CLIP, Flamingo, VQ-VAE, and the spectrum of multimodal approaches. | 🟡 |
| [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | Chain-of-thought, tree-of-thought, self-consistency, ReAct, and the full prompt-engineering toolkit, rigorously surveyed. | 🟡 |

---

## Distill: interactive deep dives

Distill.pub published the highest-quality ML explanations until 2021. All articles are interactive and mathematically rigorous.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) | The original interactive attention explainer — NTM, attention, and memory. The conceptual ancestor of the transformer. | 🟡 |
| [Why Momentum Really Works](https://distill.pub/2017/momentum/) | The real math behind momentum and acceleration — eigenvalue analysis of the convergence rate. Ties to optimization theory. | 🟡 |
| [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) | Interactive exploration of t-SNE's hyperparameters and failure modes. Critical for interpreting any dimensionality reduction visualization. | 🟢 |
| [Feature-wise Transformations](https://distill.pub/2018/feature-wise-transformations/) | A general framework for conditioning one network on another — FiLM and its applications. | 🟡 |

---

## Course notes (free, online)

| Resource | What it covers | Level |
|----------|---------------|-------|
| [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/) | Top-down practical approach: build first, understand later. The best way to get code running fast. | 🟢 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer) | Attention Is All You Need annotated line-by-line with working PyTorch code. Read after Jalammar. | 🟡 |
| [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf) | Classical ML fundamentals from Andrew Ng — linear/logistic regression, SVMs, EM, Bayesian methods. The rigorous foundation. | 🟡 |
| [Stanford CS231n](https://cs231n.stanford.edu/) | CNNs for visual recognition — image classification, detection, segmentation. The complete vision DL course. | 🟡 |
| [Stanford CS224n](https://web.stanford.edu/class/cs224n/) | NLP with Deep Learning — from word vectors through transformers and LLMs. The complete NLP course. | 🟡 |
