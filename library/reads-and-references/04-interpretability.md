# Interpretability

The full Anthropic circuits canon plus the Distill interpretability work — the resources for understanding what's actually happening inside neural networks. Curated and current as of mid-2026.

For the underlying concepts (superposition, SAEs, features, circuits), see the concept-library index: [`../bricks/README.md`](../bricks/README.md).

---

## Start here

New to the field? Read these three, in this order, before anything else.

- **Mechanistic Interpretability Essay** — Neel Nanda's accessible overview of what mech interp is and why it matters. The fastest orientation. https://transformer-circuits.pub/2022/mech-interp-essay/index.html
- **A Mathematical Framework for Transformer Circuits** — the foundational vocabulary: residual stream, attention heads as low-rank maps, virtual weights. Everything below speaks this language. https://transformer-circuits.pub/2021/framework/index.html
- **Toy Models of Superposition** — why features superpose (more features than neurons) and why sparse dictionaries are needed to recover them. The theoretical spine of modern interp. https://transformer-circuits.pub/2022/toy_model/index.html

---

## Key papers

### The Anthropic circuits program (read in order)

A coherent, chronological research program — each paper builds on the last.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | The foundational paper. Residual stream view, attention heads as low-rank matrices, virtual weights. | 🔴 |
| [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Induction heads as the discovered mechanism behind ICL — a two-layer "copy previous context" circuit. The first major circuit finding. | 🔴 |
| [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) | Why features superpose, stored as linear combinations. The theoretical foundation for why SAEs are needed. | 🔴 |
| [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | Sparse autoencoders on a one-layer MLP — extracting monosemantic features. The SAE methodology paper. | 🔴 |
| [Toy Models: Superposition, Memorization, and Double Descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html) | Connects superposition, memorization, and double descent. Ties interpretability to generalization theory. | 🔴 |
| [Privileged Bases in the Transformer Residual Stream](https://transformer-circuits.pub/2023/privileged-basis/index.html) | Why neurons (not just directions) are special — changes how you read the residual stream. | 🔴 |
| [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | SAEs at production scale on Claude 3 Sonnet — millions of interpretable features, multimodal and abstract concepts. The flagship result. | 🔴 |
| [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html) | Extends SAEs across layers and model versions — how features evolve and persist. | 🔴 |
| [Circuit Tracing: Attribution Methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | The methods behind attribution graphs — how to trace circuits at scale via replacement models. | 🔴 |
| [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Attribution graphs tracing real circuits in Claude — planning, multi-step reasoning, multilingual features. The most frontier circuits work. | 🔴 |

**Most recent from Anthropic (2025–2026):**
- [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html) — evidence that models can detect and report on aspects of their own internal states.
- [Emotion Concepts in LLMs](https://transformer-circuits.pub/2026/emotions/index.html) — how emotional representations are structured in language models.

### Beyond the SAE: methods and critiques (2024–2026)

The field matured fast after Scaling Monosemanticity. These set the current methodological frontier — confirm the venue/authors before citing, as several began as preprints.

- **Gated SAEs** (DeepMind, 2024) — separates the "which features fire" decision from "how much," cutting the shrinkage bias of L1-penalized SAEs. arXiv ID not verified here; search "Improving Dictionary Learning with Gated Sparse Autoencoders."
- **JumpReLU SAEs** (DeepMind, 2024) — a threshold activation that pushes the reconstruction/sparsity Pareto frontier past Gated and TopK variants. Verify the arXiv ID before citing.
- **TopK / BatchTopK SAEs** (OpenAI, 2024) — fixed-sparsity dictionaries from "Scaling and evaluating sparse autoencoders," with cleaner sparsity control and scaling laws for SAE width.
- **Transcoders / cross-layer transcoders** — replace an MLP block with an interpretable sparse map, enabling input-independent circuit analysis; the cross-layer variant underpins Anthropic's 2025 attribution graphs.
- **Are SAEs the right abstraction? (critiques)** — a 2024–2025 thread of work questioning whether SAE features are causal/complete, including feature-absorption and dark-matter analyses. Read alongside the methods above so you don't over-trust dictionaries. Name the specific paper from a current search rather than guessing a URL.

---

## Blogs & explainers

### Distill: feature visualization and activation atlases

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Feature Visualization](https://distill.pub/2017/feature-visualization/) | Optimization in image space to maximally activate a unit. The foundational visualization technique. | 🟡 |
| [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) | Feature visualization + attribution + semantic dictionaries combined into a CNN interpretability toolkit. | 🟡 |
| [Activation Atlas](https://distill.pub/2019/activation-atlas/) | A map of InceptionV1's feature space — what the network learned, visualized at scale. | 🟡 |
| [Circuits — Main Thread](https://distill.pub/2020/circuits/) | The original circuits program: curve detectors, high-low frequency detectors, multimodal vision neurons. Precursor to transformer circuits. | 🟡 |
| [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/) | CLIP neurons that fire for one concept across text, images, and drawings. Direct evidence of abstract representation; connects to the SAE work. | 🟡 |

### Colah's interpretability roots

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) | The geometric view of what networks learn — connects to how features organize in activation space. | 🟡 |
| [Visualizing Representations](https://colah.github.io/posts/2015-01-Visualizing-Representations/) | Early visualization of learned representations — the empirical seed of circuits research. | 🟢 |

### Walkthroughs & visual explainers

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Finding the Words to Say: Hidden State Visualizations](https://jalammar.github.io/hidden-states/) | Visualizing what LLMs store in hidden states during generation. Accessible bridge to circuits work. | 🟡 |
| [Interfaces for Explaining Transformer Language Models](https://jalammar.github.io/explaining-transformers/) | Attention visualization and explanation interfaces. Practical interpretability tooling. | 🟡 |
| Neuronpedia feature dashboards | The community hub for browsing SAE features, steering, and attribution graphs across open models — the fastest way to *see* features without running anything. https://www.neuronpedia.org |
| Neel Nanda's "Concrete Open Problems in Mech Interp" | A living list of tractable research directions; the standard on-ramp for picking a first project. Find the current version on the alignmentforum / Neel Nanda's blog. |

---

## Courses & talks

- **ARENA (Alignment Research Engineering Accelerator)** — the de facto hands-on mech interp curriculum: TransformerLens, induction heads, SAEs, and circuit-finding exercises, all in runnable notebooks. The single best way to build skills. https://www.arena.education
- **Neel Nanda — mech interp explainer videos & glossary** — paper walkthroughs and a comprehensive glossary; pair with ARENA. Find the current playlist on his YouTube channel and the glossary on his blog.
- **Anthropic interpretability team talks (2024–2026)** — conference and lab talks accompanying Scaling Monosemanticity and the attribution-graphs work; search for the most recent recordings rather than an older fixed link.

---

## Tools

- **TransformerLens** — the standard library for mechanistic interpretability on open transformers: clean access to activations, hooks, and ablations. What most papers and ARENA build on. https://github.com/TransformerLensOrg/TransformerLens
- **SAELens** — train, load, and analyze sparse autoencoders (Gated, JumpReLU, TopK) with a large library of pretrained SAEs. The practical complement to TransformerLens. https://github.com/jbloomAus/SAELens
- **nnsight / NDIF** — run interventions and read internals on very large hosted models without local weights; the way to do interp on frontier-scale open models. https://github.com/ndif-team/nnsight
- **Neuronpedia** — interactive feature explorer, steering, and attribution-graph viewer; also exposes an API over open-model SAE features. https://www.neuronpedia.org
- **pyvene** — a configurable library for activation interventions and causal-abstraction experiments across architectures. https://github.com/stanfordnlp/pyvene
