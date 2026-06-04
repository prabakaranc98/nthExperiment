# 09 · Tabular Foundation Models

*Pretrain on synthetic worlds, predict by in-context Bayesian inference — one forward pass, no per-table training.*

## The adaptation
- **Token:** a single cell, embedded with its column context; a row is a set of cells, a table is a set of rows. Order-invariance, not sequence order, is the prior.
- **Objective:** fit a *prior* over data-generating processes (TabPFN samples structural causal models / Bayesian nets), then train to predict held-out targets in-context. The network *approximates posterior predictive inference*.
- **Inductive bias:** column-then-row attention; permutation invariance over rows and columns; robustness to heterogeneous, mixed-type, missing features.
- **Verification signal:** held-out accuracy and calibration on real tables — benchmarked against tuned GBDTs, now tracked on the living [TabArena](https://arxiv.org/abs/2506.16791) leaderboard.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| FT-Transformer | Yandex | 2021 | [arXiv 2106.11959](https://arxiv.org/abs/2106.11959) | Feature tokenization + attention; the pre-FM transformer baseline for tables |
| Tree-based models still outperform DL | Grinsztajn et al. | 2022 | [arXiv 2207.08815](https://arxiv.org/abs/2207.08815) | The skeptic's benchmark — the bar every TFM must clear |
| TabPFN | Prior Labs / Freiburg | 2022 | [arXiv 2207.01848](https://arxiv.org/abs/2207.01848) | Cracked the domain: SCM-prior synthetic pretraining + in-context Bayesian prediction, one forward pass |
| TabDPT | Layer 6 AI | 2024 | [arXiv 2410.18164](https://arxiv.org/abs/2410.18164) | ICL pretrained on *real* tables + retrieval; demonstrates power-law scaling |
| CARTE | Soda / Inria | 2024 | [arXiv 2402.16785](https://arxiv.org/abs/2402.16785) | Rows as star graphs + string embeddings → transfer across tables with unmatched columns |
| TabPFN v2 | Prior Labs | 2025 | [Nature](https://www.nature.com/articles/s41586-024-08328-6) | Mixed types, regression, ~10k rows / 500 features; beats 4-hour tuned ensembles in seconds |
| TabICL | Soda / Inria | 2025 | [arXiv 2502.05564](https://arxiv.org/abs/2502.05564) | Column-then-row attention scales ICL to ~500k samples, up to 10× faster than TabPFN v2 |
| ConTextTab | SAP | 2025 | [arXiv 2506.10707](https://arxiv.org/abs/2506.10707) | Semantics-aware, table-native ICL; uses column *meaning*, strong on the CARTE benchmark |
| TabArena | AutoGluon team | 2025 | [arXiv 2506.16791](https://arxiv.org/abs/2506.16791) | Living, continuously maintained leaderboard; ends the stale-benchmark era |
| TabPFN-2.5 | Prior Labs | 2025 | [arXiv 2511.08667](https://arxiv.org/abs/2511.08667) | 50k rows / 2k features; current TabArena SOTA, matches tuned AutoGluon; distills to a fast MLP/tree |

## Where it stands (2025-2026)
- **TabPFN-2.5** leads TabArena, beating tuned tree ensembles and *matching* a 4-hour AutoGluon stack — 100% win rate vs. default XGBoost on ≤10k-row classification. A distillation engine collapses it into a low-latency MLP or tree for deployment.
- **Scaling is solved for medium data.** TabICL and TabFlex (linear attention) push ICL toward million-row tables; the small-data-only critique no longer holds.
- **Semantics are the new axis.** ConTextTab and similar models read column *names and values* as text, narrowing the gap on semantically rich tables where blind SCM priors fall short.
- **Benchmarking matured.** TabArena's living leaderboard and TALENT replaced fragmented static suites; the consensus: TFMs win on small/medium data, GBDTs stay competitive at scale, and ensembling both is SOTA.

## Transferred vs. reinvented
**Transferred from language modeling**
- The transformer block and attention.
- In-context learning as the inference mechanism — examples in the "prompt," prediction without weight updates.
- Foundation-model framing: pretrain once, deploy zero-shot across tasks.

**Reinvented for tables**
- The pretraining corpus: *synthetic* datasets sampled from a prior over causal/Bayesian generative processes, not scraped text.
- Permutation-invariant column-then-row attention instead of causal sequence ordering.
- The objective as explicit posterior predictive (Bayesian) inference, not next-token prediction.
- Native handling of mixed types, missing values, and heterogeneous schemas.

## Open problems
- **High-dimensional / wide tables and very large row counts** still strain the in-context paradigm despite TabICL/TabPFN-2.5 gains.
- **Regression and calibration** lag classification; uncertainty quality under distribution shift is uneven.
- **Synthetic prior vs. real-world fit** — does the SCM prior match deployment distributions, and how much does semantic (text) grounding help?
- **Interpretability and causality:** the SCM prior implies the representations encode causal structure; probing TFMs for causal graphs is an open, active thread.

## See also
- [03 · Time Series](../03-time-series/README.md) — the other "structured numeric" domain reaching for zero-shot foundation models.
- [11 · Design Patterns](../11-design-patterns/README.md) — in-context learning and synthetic-prior pretraining as cross-domain recipes.
- [12 · Virtual Cells](../12-virtual-cells/README.md) — high-dimensional heterogeneous measurements, a sibling tabular-style challenge.
- [Concept Library](../../../library/bricks/README.md) — conformal prediction, scaling laws, and other reusable bricks.
