# Theory & Foundations

NTK, double descent, feature learning theory, scaling laws — the rigorous explanations for why deep learning works.

Concept index: [`../bricks/README.md`](../bricks/README.md).

Level key: 🟢 accessible · 🟡 intermediate · 🔴 technical.

---

## Start here

Two pieces that frame the whole field and tell you where it stands in 2026.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [There Will Be a Scientific Theory of Deep Learning (Simon et al., 2026)](https://arxiv.org/abs/2604.21691) | The state-of-the-field manifesto. Argues DL is becoming a predictive science ("learning mechanics") and organizes the evidence into five threads: solvable settings, tractable limits, macroscopic laws, hyperparameter theories, and universal behaviors. Read this first to get the map. | 🟡 |
| [Some Math behind Neural Tangent Kernel (Lilian Weng)](https://lilianweng.github.io/posts/2022-09-08-ntk/) | The clearest survey-level NTK derivation: what it means for training dynamics, and when it is a good vs. bad approximation. The reference point everything else departs from. | 🔴 |

---

## Key papers

The empirical and theoretical results that define the field. Roughly chronological within each cluster.

### Scaling laws

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) | The original power laws — loss as a function of compute, data, and parameters. The empirical science that defines modern LLM training. | 🔴 |
| [Training Compute-Optimal LLMs / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) | The compute-optimal correction — train on more data, not just bigger models. The result that changed how frontier labs allocate compute. | 🔴 |
| [How Feature Learning Can Improve Neural Scaling Laws (Bordelon et al., 2024)](https://arxiv.org/abs/2409.17858) | A solvable model beyond the kernel limit. Feature learning in the mean-field regime derives the scaling exponents and explains why real networks beat NTK predictions — nearly doubling the compute exponent on hard tasks. | 🔴 |

### Generalization and overparameterization

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Understanding Deep Learning Requires Rethinking Generalization (Zhang et al., 2017)](https://arxiv.org/abs/1611.03530) | The memorization experiment that broke classical theory. Networks fit random labels yet generalize on real data. | 🔴 |
| [Reconciling Modern ML and the Bias-Variance Tradeoff (Belkin et al., 2018)](https://arxiv.org/abs/1812.11118) | The double-descent paper. The risk curve continues past the interpolation threshold into a second descent. | 🔴 |
| [Grokking (Power et al., 2022)](https://arxiv.org/abs/2201.02177) | Delayed generalization in transformers — the training-dynamics anomaly that exposed how circuits form long after the loss plateaus. | 🔴 |

### Feature learning and infinite-width limits

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Tensor Programs V: Zero-Shot Hyperparameter Transfer / μP (Yang et al., 2022)](https://arxiv.org/abs/2203.03466) | Maximal Update Parametrization (μP): keep every layer learning features as width → ∞, so optimal hyperparameters become width-invariant. You tune a small model and transfer ("μTransfer") to the large one. The theory practitioners actually use to scale. | 🔴 |

### Interpretability as evidence for theory

| Resource | Why read it | Level |
|----------|-------------|-------|
| [A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html) | Reverse-engineers attention-only transformers; introduces induction heads and the QK/OV decomposition. The foundation of mechanistic interpretability and a concrete instance of "we can actually understand what these weights compute." | 🔴 |
| [Toy Models of Superposition (Elhage et al., 2022)](https://transformer-circuits.pub/2022/toy_model/index.html) | Why networks pack more features than they have neurons (superposition), and when polysemanticity is fully explainable. A clean solvable model with broad implications for feature learning. | 🟡 |

---

## Blogs & explainers

Timeless visual and conceptual pieces. These age well — keep them.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Colah: Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | Entropy, KL divergence, cross-entropy — visualized. The information-theoretic view of what training minimizes. | 🟢 |
| [Colah: Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | Why deep nets can classify: they learn homeomorphisms that unentangle data manifolds. The geometric view of representation. | 🟡 |
| [Colah: Neural Networks, Types, and Functional Programming](https://colah.github.io/posts/2015-09-NN-Types-FP/) | The typed perspective — layer type signatures and compositionality. Unusual and illuminating. | 🟡 |
| [Distill: Why Momentum Really Works](https://distill.pub/2017/momentum/) | The correct explanation — eigenvalue analysis, not "rolling downhill." Connects to NTK convergence. | 🟡 |
| [Distill: Adversarial Examples Are Not Bugs, They Are Features (discussion)](https://distill.pub/2019/advex-bugs-discussion/) | The debate over whether adversarial examples reveal something real about learned features. Connects to feature-learning theory. | 🟡 |
| [Distill: Research Debt](https://distill.pub/2017/research-debt/) | The meta-problem: research accumulates faster than it can be absorbed. The case for distillation and clear exposition. | 🟢 |

---

## Courses & talks

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Physics of Language Models (Allen-Zhu) — ICML 2024 tutorial + project page](https://physics.allen-zhu.com/) | A controlled-experiment program: train LLMs on synthetic data to isolate universal laws of knowledge storage, reasoning, and structure. The cleanest case for treating LLMs as a physical system. Tutorial video and parts are linked from the page. | 🟡 |
| Boris Hanin — *Lectures on Deep Learning Theory* (Princeton ORFE, 2024 lecture notes) | Self-contained notes on NTK regimes, large-width behavior, and training dynamics. A rigorous entry point for the infinite-width line of work. Search the author's Princeton page for the current PDF. | 🔴 |

---

## Tools

| Resource | What it is | Level |
|----------|-----------|-------|
| [`mup` (microsoft/mup)](https://github.com/microsoft/mup) | Reference PyTorch implementation of Maximal Update Parametrization. Wrap a model, set base shapes, and get width-invariant hyperparameters — the practical companion to Tensor Programs V. | 🔴 |

---

*See also the concept library index: [`../bricks/README.md`](../bricks/README.md).*
