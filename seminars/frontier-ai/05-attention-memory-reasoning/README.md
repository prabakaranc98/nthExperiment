# Topic 05 · Attention, Memory, Reasoning & Sequence Models

*The transformer's internals, the sub-quadratic alternatives, and the shift from single-pass to search/verify/test-time-compute reasoning.*

**Papers:** 20 · **Pace:** ~10 days at 2/day

---

## Paper Log

| # | Paper | Authors | Year | Status | Note | Blog | Exp |
|---|-------|---------|------|--------|------|------|-----|
| 1 | NMT by Jointly Learning to Align and Translate | Bahdanau et al. | 2014 | queued | The attention mechanism's origin | — | — |
| 2 | On Layer Normalization in the Transformer (Pre-LN) | Xiong et al. | 2020 | queued | Norm placement decides training stability | — | — |
| 3 | RMSNorm | Zhang & Sennrich | 2019 | queued | The norm modern LLMs actually use | — | — |
| 4 | GLU Variants (SwiGLU) | Shazeer | 2020 | queued | The FFN that became standard | — | — |
| 5 | RoFormer (RoPE) | Su et al. | 2021 | queued | Rotary positions; encoding under long-context models | — | — |
| 6 | Longformer | Beltagy et al. | 2020 | queued | Sparse attention for linear-scaling context | — | — |
| 7 | Ring Attention | Liu, Zaharia & Abbeel | 2023 | queued | Context length scaled by sharding attention across devices | — | — |
| 8 | Structured State Spaces (S4) | Gu, Goel & Ré | 2021 | queued | Sub-quadratic sequence modeling from linear dynamical system | — | — |
| 9 | Mamba | Gu & Dao | 2023 | queued | Selective SSMs; content-based reasoning without attention | — | — |
| 10 | Transformers are SSMs (Mamba-2) | Dao & Gu | 2024 | queued | Duality unifying attention and state-space models | — | — |
| 11 | Hopfield Networks Is All You Need | Ramsauer et al. | 2020 | queued | Modern continuous Hopfield memory ≈ attention | — | — |
| 12 | Chain-of-Thought Prompting | Wei et al. | 2022 | queued | Intermediate computation unlocks multi-step reasoning | — | — |
| 13 | Self-Consistency | Wang et al. | 2022 | queued | Marginalizing over reasoning paths — sampling estimator | — | — |
| 14 | Tree of Thoughts | Yao et al. | 2023 | queued | Search over reasoning states | — | — |
| 15 | Let's Verify Step by Step | Lightman et al. | 2023 | queued | Process reward models — supervising the reasoning | — | — |
| 16 | Scaling LLM Test-Time Compute Optimally | Snell et al. | 2024 | queued | Inference-time compute traded against parameters | — | — |
| 17 | Retrieval-Augmented Generation (RAG) | Lewis et al. | 2020 | queued | Parametric memory + non-parametric retriever | — | — |
| 18 | RETRO | Borgeaud et al. | 2021 | queued | Retrieval baked into pretraining | — | — |
| 19 | Overcoming Catastrophic Forgetting (EWC) | Kirkpatrick et al. | 2017 | queued | Forgetting as Fisher-weighted parameter drift | — | — |
| 20 | iCaRL | Rebuffi et al. | 2017 | queued | Class-incremental learning with exemplar rehearsal | — | — |

---

## Synthesis Notes
## Blog Post
## Experiments
