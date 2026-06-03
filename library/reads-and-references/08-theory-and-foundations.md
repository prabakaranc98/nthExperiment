# Theory & Foundations

NTK, double descent, feature learning theory, scaling laws — the rigorous explanations for why deep learning works.

---

## The NTK and lazy vs. feature learning (Lilian Weng)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Some Math behind Neural Tangent Kernel](https://lilianweng.github.io/posts/2022-09-08-ntk/) | The NTK derivation, what it means for training dynamics, and when it's a good vs. bad approximation. The clearest survey-level NTK explanation. | 🔴 |

---

## Distill: optimization theory

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Why Momentum Really Works](https://distill.pub/2017/momentum/) | The correct explanation — eigenvalue analysis, not intuition about "rolling downhill." Connects to NTK convergence analysis. | 🟡 |
| [Research Debt](https://distill.pub/2017/research-debt/) | The meta-problem: ML research accumulates faster than it can be absorbed. The argument for why distillation and clear exposition matter. | 🟢 |

---

## Information theory and representations

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Colah: Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | Entropy, KL divergence, cross-entropy — visualized. The information-theoretic view of what training minimizes. | 🟢 |
| [Colah: Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | Why deep networks can solve classification: they learn homeomorphisms that unentangle data manifolds. The geometric view of representation. | 🟡 |
| [Colah: Neural Networks, Types, and Functional Programming](https://colah.github.io/posts/2015-09-NN-Types-FP/) | The typed perspective on neural networks — type signatures of layers, compositionality. Unusual and illuminating. | 🟡 |

---

## Scaling laws

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Scaling Laws for Neural Language Models (Kaplan et al.)](https://arxiv.org/abs/2001.08361) | The original power laws — loss as a function of compute, data, and parameters. The empirical science that defines modern LLM training. | 🔴 |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | The compute-optimal correction — you should train on more data, not just bigger models. The result that changed how frontier labs train. | 🔴 |

---

## Generalization and overparameterization

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Understanding Deep Learning Requires Rethinking Generalization (Zhang et al.)](https://arxiv.org/abs/1611.03530) | The memorization experiment that broke classical theory. Networks memorize random labels yet generalize on real data. | 🔴 |
| [Reconciling Modern ML and the Bias-Variance Tradeoff (Belkin et al.)](https://arxiv.org/abs/1812.11118) | The double descent paper. The U-curve continues past the interpolation threshold into a second descent. | 🔴 |
| [Grokking (Power et al.)](https://arxiv.org/abs/2201.02177) | Delayed generalization in transformers — the training dynamics anomaly that revealed how circuits form. | 🔴 |

---

## Distill: adversarial examples and robustness

| Resource | Why read it | Level |
|----------|-------------|-------|
| [A Discussion of Adversarial Examples Are Not Bugs, They Are Features](https://distill.pub/2019/advex-bugs-discussion/) | The debate over whether adversarial examples reveal something deep about what neural networks learn. Connects to feature learning theory. | 🟡 |

---

## The emerging science of deep learning

| Resource | Why read it | Level |
|----------|-------------|-------|
| [How Feature Learning Can Improve Neural Scaling Laws (Bordelon et al.)](https://arxiv.org/abs/2409.17858) | Why real networks outperform NTK predictions — feature learning in the mean-field regime derives the scaling law exponents. | 🔴 |
| [There Will Be a Scientific Theory of Deep Learning (Simon et al.)](https://arxiv.org/abs/2604.21691) | The manifesto — what a rigorous, predictive DL science requires and where the field currently stands. | 🔴 |
