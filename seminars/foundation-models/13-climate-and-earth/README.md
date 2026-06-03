# 13 · Climate & Earth Foundation Models

*Weather forecasting beating numerical models. Earth system emulation. The verification-rich domain.*

**Why this domain is special:** weather forecasting has perfect ground truth (tomorrow's weather) and massive historical data (reanalysis datasets). It's one of the first domains where FMs definitively outperformed expert-built numerical models on operational metrics.

**The "token":** a spatiotemporal grid cell at a pressure level. The pretraining objective: predict the next atmospheric state.

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Pangu-Weather | Bi et al., Huawei | 2023 | [Nature 2023](https://www.nature.com/articles/s41586-023-06185-3) | First FM to beat ECMWF on medium-range forecast |
| FourCastNet | Pathak et al., NVIDIA | 2022 | [arXiv 2202.11214](https://arxiv.org/abs/2202.11214) | Adaptive Fourier neural operator for weather |
| GraphCast | Lam et al., DeepMind | 2023 | [Science 2023](https://www.science.org/doi/10.1126/science.adi2336) | GNN-based; outperforms HRES on 10-day forecast |
| ClimaX | Nguyen et al. | 2023 | [arXiv 2301.10343](https://arxiv.org/abs/2301.10343) | Foundation model for weather and climate; ViT backbone |
| Aurora | Bodnar et al., Microsoft | 2024 | [arXiv 2405.13063](https://arxiv.org/abs/2405.13063) | 1B parameter FM for atmospheric forecasting |
| AIFS (AI Integrated Forecasting System) | ECMWF | 2024 | [arXiv 2406.01465](https://arxiv.org/abs/2406.01465) | Official ECMWF operational ML forecast model |
| Prithvi-WxC | NASA / IBM | 2024 | [arXiv 2409.13598](https://arxiv.org/abs/2409.13598) | NASA earth observation foundation model |
| SatVAE / Clay | Clay Foundation | 2024 | [GitHub](https://github.com/Clay-foundation/model) | Foundation model for satellite imagery |

**The scaling result:** weather FMs trained on ERA5 reanalysis data generalize to operational forecasting better than physics-based models at a fraction of the compute cost.
**Open question:** Can these models extrapolate to out-of-distribution climate states (future warming scenarios) or do they only interpolate within historical distribution?
