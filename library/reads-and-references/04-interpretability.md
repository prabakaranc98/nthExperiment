# Interpretability

The full Anthropic circuits canon plus the Distill interpretability work — the resources for understanding what's actually happening inside neural networks.

---

## The Anthropic circuits program (read in order)

This is a coherent research program. Read chronologically — each paper builds on the last.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | The foundational paper. Introduces the residual stream view, attention heads as low-rank matrices, virtual weights. The language all subsequent circuits work uses. | 🔴 |
| [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Induction heads as the discovered mechanism behind ICL — a specific two-layer circuit that implements "copy previous context." The first major circuit finding. | 🔴 |
| [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) | Why features superpose: more features than neurons, stored as linear combinations. The theoretical foundation for why SAEs are needed. Essential. | 🔴 |
| [Mechanistic Interpretability Essay](https://transformer-circuits.pub/2022/mech-interp-essay/index.html) | Neel Nanda's essay on what mechanistic interpretability is and why it matters. Accessible overview of the program. | 🟡 |
| [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | Sparse autoencoders applied to a one-layer MLP — extracting monosemantic features. The SAE methodology paper. | 🔴 |
| [Toy Models: Superposition, Memorization, and Double Descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html) | The connection between superposition, memorization, and the double descent phenomenon. Ties interpretability to generalization theory. | 🔴 |
| [Privileged Bases in the Transformer Residual Stream](https://transformer-circuits.pub/2023/privileged-basis/index.html) | Why neurons (not just directions) are special — the empirical finding that changes how you think about the residual stream. | 🔴 |
| [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | SAEs at production scale on Claude 3 Sonnet — millions of interpretable features, multimodal neurons, abstract concepts. The flagship result. | 🔴 |
| [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html) | Extending SAEs across layers and model versions — understanding how features evolve. | 🔴 |
| [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Attribution graphs tracing actual computational circuits in Claude — how the model plans, reasons, and uses features. The most frontier work. | 🔴 |
| [Circuit Tracing: Attribution Methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | The methods paper behind the biology post — how to do circuit tracing at scale. | 🔴 |

**Also from 2026 (very recent):**
- [Emotion Concepts in LLMs](https://transformer-circuits.pub/2026/emotions/index.html) — emotional representations in language models
- [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html) — evidence of self-modeling

---

## Distill: feature visualization and activation atlases

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Feature Visualization](https://distill.pub/2017/feature-visualization/) | How to visualize what neurons detect — optimization in image space to maximally activate a unit. The foundational technique. | 🟡 |
| [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) | Combining feature visualization, attribution, and semantic dictionaries into an interpretability toolkit for CNNs. | 🟡 |
| [Activation Atlas](https://distill.pub/2019/activation-atlas/) | A map of the feature space of InceptionV1 — what the network has learned to recognize, visualized at scale. | 🟡 |
| [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/) | Neurons in CLIP that respond to the same concept across text, images, and drawings — direct evidence of abstract representation. Connects to the SAE work. | 🟡 |

---

## The Distill circuits thread (original)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Circuits — Main Thread](https://distill.pub/2020/circuits/) | The original circuits research program — curve detectors, high-low frequency detectors, multimodal neurons in vision models. The precursor to transformer circuits work. | 🟡 |

---

## Colah's interpretability roots

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | The geometric view of what neural networks learn — directly connects to how features are organized in activation space. | 🟡 |
| [Visualizing Representations](https://colah.github.io/posts/2015-01-Visualizing-Representations/) | Early work on visualizing learned representations — the empirical observation that inspired circuits research. | 🟢 |

---

## Finding hidden states

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Finding the Words to Say: Hidden State Visualizations](https://jalammar.github.io/hidden-states/) | Visualizing what LLMs store in their hidden states during generation. Accessible bridge to the circuits work. | 🟡 |
| [Interfaces for Explaining Transformer Language Models](https://jalammar.github.io/explaining-transformers/) | Attention visualization and explanation interfaces. Practical interpretability tooling. | 🟡 |
