# Frontiers in Deep Learning — A Survey & Map

*Doc 2 of 4. This one covers **deep learning specifically** — neural-network architectures, generative modeling, the science of deep nets, mechanistic interpretability, and neural networks for science. Broader theory of learning lives in Frontiers in ML; applied/decision uses live in Frontiers in Data Science.*

---

## What "frontier" means in DL

The frontier splits cleanly across four layers:

| Layer | The DL question | Live edge |
|---|---|---|
| **Architecture** | What network *shape* learns best/cheapest? | SSMs, hybrids, beyond-attention |
| **Objective** | What *training signal* yields good representations/samples? | Flow matching, JEPA |
| **Science/theory** | *Why* do deep nets learn and generalize? | Feature-learning theory, interpretability |
| **Application** | What *domains* fall to deep nets? | Proteins, materials, PDEs |

DL's defining trait: it works *spectacularly* while being *poorly understood* — so its theory/interpretability frontier is unusually wide open and unusually tractable.

---

## 1. Architecture — is attention necessary, or just sufficient?

- **Mixture-of-Experts as default** — fine-grained + shared experts, aux-loss-free balancing, MLA for KV-cache. (Full detail in the models survey.)
- **State-space models / Mamba** ([2312.00752](https://arxiv.org/abs/2312.00752)) — sub-quadratic, selective-scan; the leading challenger for long context.
- **Linear & hybrid attention** — Gated Delta Net and attention+SSM stacks that mix quality with efficiency.
- **Test-time training / adaptive computation** — layers that learn at inference; thinking budgets.
- **Long-context mechanisms** — RoPE + extensions (YaRN, position interpolation), 1M–10M token windows.

**Open problem:** An architecture matching transformer quality at linear cost that trains stably at frontier scale.

---

## 2. Generative modeling — the diffusion → flow-matching shift

- **Diffusion** (DDPM, score-based SDEs) gave stable training and SOTA images/audio/video by learning to denoise.
- **Flow matching** ([2210.02747](https://arxiv.org/abs/2210.02747)) — simulation-free training of continuous flows between arbitrary distributions; faster training and sampling, OT paths, unifiable with diffusion under a *generator-matching* view. Tutorial: [2506.02070](https://arxiv.org/abs/2506.02070).
- **Few-step generation** — consistency models and distillation collapsing sampling cost to 1–4 steps.
- **Discrete diffusion for language** — diffusion/flow over tokens as a non-autoregressive alternative; an active and surprising edge.
- **Multimodal & world simulators** — diffusion/flow for video and interactive learned environments.

**Open problem:** A single generative framework best across modalities, fast to sample, and likelihood-principled.

---

## 3. Representation & self-supervised learning

- **Contrastive** (SimCLR, CLIP) — pull positives together, push negatives apart.
- **Masked modeling** (MAE) — reconstruct masked inputs; the BERT idea generalized to vision.
- **Self-distillation** (DINO / DINOv2) — student-teacher with no labels.
- **JEPA** (I-JEPA, V-JEPA) — predict in *representation space* rather than pixel/token space; Joint-Embedding Predictive Architectures as a non-generative path to world models.

**Open problem:** Which objective yields the most transferable representation — generative, contrastive, or latent-predictive?

---

## 4. The science & theory of deep learning

The most exciting frontier because it's both deeply open and newly tractable.

- **Feature learning vs. NTK** — the central axis. NTK explains wide nets but can't explain real generalization or scaling. The frontier is the **feature-learning regime** (μP / Tensor Programs; Bordelon–Pehlevan), deriving scaling-law exponents from first principles. → [2409.17858](https://arxiv.org/abs/2409.17858)
- **Scaling-law theory** — moving from empirical Chinchilla curves to *mechanistic* explanations of why the exponents are what they are.
- **Generalization for deep nets** — NTK-eigenstructure theories unifying benign overfitting, double descent, implicit bias, and grokking.
- **Emergence & phase transitions** — real phase changes or metric artifacts? Grokking as delayed generalization with identifiable circuit formation. Allen-Zhu's "physics of language models."
- **The broader program** — a rigorous, quantitative, predictive theory of deep learning is actively emerging. → [2604.21691](https://arxiv.org/abs/2604.21691)

**Open problem:** Given (data, architecture, optimizer, scale), *predict* what features form, when, and how well it generalizes — before training.

---

## 5. Mechanistic interpretability — the science of what's inside

The most empirically tractable theory frontier.

- **Superposition** — features outnumber neurons and are stored as linear combinations (Toy Models of Superposition).
- **Sparse autoencoders (SAEs)** — extract monosemantic, interpretable features from production-scale models (Anthropic's Scaling Monosemanticity, 2024).
- **Circuits** — identifiable computational subgraphs; **induction heads** as the discovered mechanism of in-context learning.
- **Causal methods** — activation patching, causal tracing, probing to test what computation a model performs.

**Open problem:** Scale interpretability from features to full circuits of frontier models and turn it into reliable steering/monitoring.

---

## 6. Geometric, equivariant & operator-learning deep nets

- **Geometric & equivariant DL** — building symmetry (E(3)/SE(3)) into the architecture; backbone of molecular and materials models (GNoME's GNNs).
- **Neural operators** — learning maps between function spaces (FNO and successors); discretization-invariant, resolution-invariant; fast PDE solvers.
- **PDE / scientific foundation models** — pretrain across PDE families (Poseidon, CoDANO); earth-system/weather emulators.

**Open problem:** A true foundation model for physics — one operator network generalizing across equations, geometries, and boundary conditions.

---

## 7. Efficiency & systems

FlashAttention-class IO-aware kernels, speculative decoding, paged-KV serving, FP8/FP4 training, training stability at scale (MuonClip). The frontier: quality-per-FLOP and quality-per-dollar. (Full detail in the models survey's systems section.)

---

## How to stay on the DL frontier

Stand where an architectural or objective paradigm is shifting — right now: post-transformer architectures, flow-matching unification, latent-predictive (JEPA) representations, and mechanistic interpretability maturing into real science.

**The immortal DL fundamentals:** linear algebra & matrix calculus; the transformer + backprop from scratch; probability and the score/diffusion math (SDEs, ODEs, change-of-variables); optimization dynamics (SGD/Adam, init, normalization); and the information-theoretic view of representation learning.

*Next: Frontiers in Data Science and Applied Statistics for the Modern AI Era.*
