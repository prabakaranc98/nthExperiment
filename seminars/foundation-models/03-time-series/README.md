# 03 · Time Series Foundation Models

*Zero-shot forecasting. Is there a GPT-3 moment for time series?*

**The key challenge:** time series are heterogeneous (different frequencies, scales, domains), non-stationary, and have complex temporal dependencies. The "token" question: a single time step? a patch? a channel?

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Chronos: Learning the Language of Time Series | Amazon Science | 2024 | [arXiv 2403.07815](https://arxiv.org/abs/2403.07815) | Tokenize values; T5-based; zero-shot |
| Unified Training of Universal Time Series Transformers (Moirai) | Salesforce AI | 2024 | [arXiv 2402.02592](https://arxiv.org/abs/2402.02592) | Any-variate, any-frequency |
| Moirai 2.0 | Salesforce | 2024 | [arXiv 2511.11698](https://arxiv.org/abs/2511.11698) | Less is more for forecasting |
| Lag-Llama | Rasul et al., ServiceNow | 2023 | [arXiv 2310.08278](https://arxiv.org/abs/2310.08278) | Probabilistic; lag features as tokens |
| MOMENT: Open Time-Series Foundation Models | – | 2024 | [arXiv 2402.03885](https://arxiv.org/abs/2402.03885) | Family of open models |
| UniTS | Harvard MIMS | 2024 | [arXiv 2403.00131](https://arxiv.org/abs/2403.00131) | Unified multi-task time series model |
| TiRex | NX-AI | 2025 | [arXiv 2505.23719](https://arxiv.org/abs/2505.23719) | Zero-shot across long and short horizons |
| Sundial | Tsinghua | 2025 | [arXiv 2502.00816](https://arxiv.org/abs/2502.00816) | Flow matching for time series |
| Timer-XL | THUML | 2024 | [arXiv 2410.04803](https://arxiv.org/abs/2410.04803) | Long-context transformers for unified forecasting |
| A decoder-only foundation model for time-series (TimesFM) | Google Research | 2024 | [ICML 2024](https://github.com/google-research/timesfm) | Patch-based; decoder-only |

**What time series had to invent:** channel-mixing vs. channel-independent strategies, frequency-agnostic tokenization, handling non-stationarity at scale.
**Open question:** Does a true "GPT-3 moment" exist for time series, or is the domain too heterogeneous for a single universal model?
