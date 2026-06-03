# 09 · Tabular Foundation Models

*In-context Bayesian prediction. The FM paradigm applied to data science's home turf.*

**The key breakthrough:** TabPFN — pretraining on synthetic datasets sampled from a prior over structural causal models, then using in-context learning at inference to do Bayesian prediction in a single forward pass. Beats tuned GBDTs on small data.

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| TabPFN | Hollmann et al., Prior Labs | 2022 | [arXiv 2207.01848](https://arxiv.org/abs/2207.01848) | In-context Bayesian prediction; beats GBDTs on small data |
| TabPFN v2 | Prior Labs | 2023 | [arXiv 2311.10609](https://arxiv.org/abs/2311.10609) | Scales to 10k rows, mixed types |
| TabPFN-2.5 | Prior Labs | 2024 | [arXiv 2511.08667](https://arxiv.org/abs/2511.08667) | 100k rows, 2k features; matches tuned ensembles |
| Revisiting Deep Learning Models for Tabular Data (FT-Transformer) | Gorishniy et al. | 2021 | [arXiv 2106.11959](https://arxiv.org/abs/2106.11959) | Feature tokenization + attention for tabular |
| SAINT | Somepalli et al. | 2021 | [arXiv 2106.01342](https://arxiv.org/abs/2106.01342) | Row attention + contrastive pretraining |
| TabNet | Arık & Pfister | 2019 | [arXiv 1908.07442](https://arxiv.org/abs/1908.07442) | Attentive sequential feature selection |
| Why tree-based models still outperform DL on tabular data | Grinsztajn et al. | 2022 | [arXiv 2207.08815](https://arxiv.org/abs/2207.08815) | Critical perspective — read alongside TabPFN |

**What tabular had to invent:** the SCM-prior synthetic pretraining idea (TabPFN), feature tokenization, handling heterogeneous column types.
**The causal connection:** TabPFN's synthetic prior is over SCMs → the representations implicitly encode causal structure. Probing TabPFN for causal graphs is an active research thread.
