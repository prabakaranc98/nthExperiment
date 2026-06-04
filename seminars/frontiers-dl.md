# Frontiers in Deep Learning — A Survey & Map

*Doc 2 of 4. This one covers **deep learning specifically** — neural-network architectures, generative modeling, the science of deep nets, mechanistic interpretability, and neural networks for science. Broader theory of learning lives in Frontiers in ML; applied/decision uses live in Frontiers in Data Science.*

> Concept cross-references point to the [concept-library index](../library/bricks/README.md).

---

## What "frontier" means in DL

The frontier splits cleanly across four layers.

| Layer | The DL question | Live edge (2025-2026) |
|---|---|---|
| **Architecture** | What network *shape* learns best/cheapest? | Hybrid attention+SSM stacks now ship in flagship models |
| **Objective** | What *training signal* yields good representations/samples? | Flow matching, latent-predictive (JEPA), discrete diffusion |
| **Science/theory** | *Why* do deep nets learn and generalize? | Feature-learning scaling theory; interpretability as science |
| **Application** | What *domains* fall to deep nets? | Proteins, materials, PDE/weather operators |

DL's defining trait: it works *spectacularly* while being *poorly understood*. Its theory and interpretability frontier is therefore unusually wide open — and, lately, unusually tractable.

---

## 1. Architecture — is attention necessary, or just sufficient?

The 2025-2026 answer is empirical: **not strictly necessary, but hard to remove cleanly.** Flagship models now interleave linear/SSM layers with a minority of full-attention layers rather than replacing attention outright.

| Direction | What it is | State in 2025-2026 |
|---|---|---|
| **MoE as default** | Fine-grained + shared experts, aux-loss-free balancing, MLA for KV cache | Standard at frontier scale (detail in the models survey) |
| **Selective SSMs / Mamba** | Sub-quadratic selective scan ([2312.00752](https://arxiv.org/abs/2312.00752)) | Mamba-2/3 used as a *component*, rarely alone |
| **Gated linear attention** | Gated DeltaNet (delta rule over a decaying matrix state) | Backbone of Qwen3-Next, Kimi Linear's KDA |
| **Hybrid attention+SSM** | Few full-attention layers among many linear layers | Qwen3-Next (~3:1), Nemotron-H, MiniMax — the dominant pattern |
| **Test-time training** | Layer state is a small model updated by online gradient at inference (TTT, Titans) | Active; expressive recurrent memory |
| **Long context** | RoPE + YaRN/position interpolation | 1M-token windows shipping; 10M claimed |

**The live debate.** Whether linear attention is production-ready is genuinely contested. Some labs commit to hybrids for throughput; others (e.g., MiniMax with M2) reverted to full attention citing weaker multi-turn and reasoning recall. The diagnostic remains associative recall vs. state size.

**Open problem:** an architecture matching transformer quality at linear cost that trains stably at frontier scale and does not lose recall — still unsolved.

---

## 2. Generative modeling — the diffusion → flow-matching shift

- **Diffusion** (DDPM, score-based SDEs) gave stable training and SOTA images/audio/video by learning to denoise.
- **Flow matching** ([2210.02747](https://arxiv.org/abs/2210.02747)) — simulation-free training of continuous flows between arbitrary distributions; faster training and sampling, OT paths, unifiable with diffusion under a *generator-matching* view. Tutorial: [2506.02070](https://arxiv.org/abs/2506.02070).
- **One/few-step generation** — the cost frontier has collapsed from "consistency distillation" to **flow-map / average-velocity** methods: MeanFlow ([2505.13447](https://arxiv.org/abs/2505.13447)), shortcut models, and policy-distillation (π-Flow) put strong text-to-image in 1-4 steps.
- **Discrete diffusion for language** — non-autoregressive token generation matured fast: DeepMind's Gemini Diffusion is the first production-grade diffusion LLM, uniform-noise models have been scaled to ~10B params, and step-budget-conditioned discrete flow matching reaches usable quality in 8-16 steps.
- **Multimodal & world simulators** — diffusion/flow for video and interactive learned environments.

**Open debate:** consistency-style models provably accumulate error over multi-step sampling, motivating flow-map distillation (e.g., Align Your Flow). No single distillation recipe yet dominates on quality, stability, and step count together.

**Open problem:** one generative framework that is best across modalities, fast to sample, and likelihood-principled.

---

## 3. Representation & self-supervised learning

| Family | Objective | Exemplars |
|---|---|---|
| **Contrastive** | Pull positives together, push negatives apart | SimCLR, CLIP |
| **Masked modeling** | Reconstruct masked inputs | MAE |
| **Self-distillation** | Student-teacher, no labels | DINO / DINOv2 |
| **JEPA (latent-predictive)** | Predict in *representation* space, not pixels/tokens | I-JEPA, V-JEPA, **V-JEPA 2** ([2506.09985](https://arxiv.org/abs/2506.09985)) |

The latent-predictive line is the live edge. V-JEPA 2 (2025) pretrains on ~1M hours of video, then post-trains an action-conditioned variant for zero-shot robot planning — pushing JEPA from a representation objective toward a usable **world model**.

**Open problem:** which objective yields the most transferable representation — generative, contrastive, or latent-predictive?

---

## 4. The science & theory of deep learning

The most exciting frontier: deeply open, yet newly tractable.

- **Feature learning vs. NTK** — the central axis. NTK explains wide nets but not real generalization or scaling. The frontier is the **feature-learning regime** (μP / Tensor Programs; Bordelon–Atanasov–Pehlevan), which derives *when* feature learning changes scaling exponents: unchanged for easy targets inside the initial-NTK RKHS, nearly doubled for hard targets outside it. → [2409.17858](https://arxiv.org/abs/2409.17858)
- **Scaling-law theory** — moving from empirical Chinchilla curves to *mechanistic* derivations of the exponents themselves.
- **Generalization for deep nets** — eigenstructure theories unifying benign overfitting, double descent, implicit bias, and grokking.
- **Emergence & phase transitions** — real phase changes or metric artifacts? Grokking as delayed generalization with identifiable circuit formation; Allen-Zhu's "physics of language models."

**Open problem:** given (data, architecture, optimizer, scale), *predict* what features form, when, and how well the model generalizes — before training.

---

## 5. Mechanistic interpretability — the science of what's inside

The most empirically tractable theory frontier, and the one that moved most in 2025.

| Tool | What it buys |
|---|---|
| **Superposition** | Features outnumber neurons, stored as near-orthogonal linear directions (Toy Models of Superposition) |
| **Sparse autoencoders (SAEs)** | Monosemantic features from production models (Anthropic, Scaling Monosemanticity, 2024) |
| **Cross-layer transcoders + attribution graphs** | Replace MLPs with sparse cross-layer modules, then trace per-prompt computation end-to-end (Anthropic, "Biology of an LLM," 2025) |
| **Circuits / induction heads** | Identifiable subgraphs; induction heads as the discovered ICL mechanism |
| **Causal methods** | Activation patching, attribution patching, causal tracing, probing |

**2025 inflection:** attribution graphs and open-sourced circuit-tracing tooling (Anthropic; the community Circuit Tracer / CLT libraries on Gemma, Llama, Qwen) turned per-prompt circuit analysis into a repeatable workflow rather than a one-off study.

**Open problem:** scale interpretability from features to full circuits of frontier models, and convert it into reliable steering and monitoring.

---

## 6. Geometric, equivariant & operator-learning deep nets

- **Geometric & equivariant DL** — bake symmetry (E(3)/SE(3)) into the architecture; the backbone of molecular and materials models (GNoME's GNNs).
- **Neural operators** — learn maps between function spaces (FNO and successors); discretization- and resolution-invariant; fast PDE solvers.
- **PDE / scientific foundation models** — pretrain across PDE families (Poseidon, [2405.19101](https://arxiv.org/abs/2405.19101); CoDANO) then fine-tune; the same recipe now anchors data-scarce regimes such as 3D atmospheric and earth-system emulation.

**Open problem:** a true foundation model for physics — one operator network generalizing across equations, geometries, and boundary conditions.

---

## 7. Efficiency & systems

| Lever | Examples |
|---|---|
| **IO-aware kernels** | FlashAttention-class attention |
| **Serving** | Speculative decoding, paged-KV, prefix caching, disaggregated prefill/decode |
| **Low precision** | FP8 training; FP4/MXFP microscaling for training and inference |
| **Stability at scale** | Muon / MuonClip optimizers, z-loss, logit soft-capping |

The frontier metric is quality-per-FLOP and quality-per-dollar. (Full detail in the models survey's systems section.)

---

## How to stay on the DL frontier

Stand where an architectural or objective paradigm is shifting. Right now:

- **Architecture** — hybrid attention+SSM stacks moving from research to flagships.
- **Objective** — flow-matching unification and one-step flow-map distillation.
- **Representation** — latent-predictive (JEPA) models becoming world models.
- **Theory** — feature-learning scaling theory and mechanistic interpretability maturing into real science.

**The immortal DL fundamentals:** linear algebra & matrix calculus; the transformer + backprop from scratch; probability and the score/diffusion math (SDEs, ODEs, change-of-variables); optimization dynamics (SGD/Adam, init, normalization); and the information-theoretic view of representation learning.

*Next: Frontiers in Data Science and Applied Statistics for the Modern AI Era.*
