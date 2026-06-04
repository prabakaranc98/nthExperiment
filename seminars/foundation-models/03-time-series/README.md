# 03 · Time Series Foundation Models
*One model, any series, no training — and the field is still arguing whether that's enough.*

## The adaptation
- **Token** — not a single timestep. The winning unit is a **patch** of contiguous values (TimesFM, Moirai) or a **quantized value bin** treated as a vocabulary (Chronos). Patching cuts sequence length and forces local structure.
- **Objective** — autoregressive next-patch prediction, mostly **decoder-only**. Probabilistic variants emit full distributions (quantiles, flow/diffusion heads) rather than point forecasts.
- **Inductive bias** — scale/frequency invariance via instance normalization, plus channel-independence or learned cross-variate attention. The series must be readable at any resolution, scale, or variate count.
- **Verification signal** — held-out, leakage-controlled zero-shot error on broad benchmarks (**GIFT-Eval**, **BOOM**) using MASE for point and CRPS for probabilistic accuracy.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| Lag-Llama | ServiceNow / Mila | 2023 | [arXiv 2310.08278](https://arxiv.org/abs/2310.08278) | First open decoder-only probabilistic forecaster; lag features as tokens |
| TimesFM | Google Research | 2023 | [arXiv 2310.10688](https://arxiv.org/abs/2310.10688) | The landmark: 200M patched decoder, 100B time points, near-SOTA zero-shot |
| Moirai | Salesforce | 2024 | [arXiv 2402.02592](https://arxiv.org/abs/2402.02592) | Any-variate, any-frequency masked encoder; built the LOTSA corpus + GIFT-Eval |
| Chronos | Amazon | 2024 | [arXiv 2403.07815](https://arxiv.org/abs/2403.07815) | Quantize values into a token vocabulary; reuse a T5 LLM stack directly |
| Toto + BOOM | Datadog | 2025 | [arXiv 2505.14766](https://arxiv.org/abs/2505.14766) | Observability-tuned multivariate model; ships the real-world BOOM benchmark |
| Sundial | Tsinghua | 2025 | [arXiv 2502.00816](https://arxiv.org/abs/2502.00816) | Generative flow-matching head; samples forecast distributions, not quantiles |
| TiRex | NX-AI | 2025 | [arXiv 2505.23719](https://arxiv.org/abs/2505.23719) | xLSTM backbone; tops GIFT-Eval across short and long horizons without attention |
| In-Context Fine-Tuning (TimesFM-ICF) | Google Research | 2025 | [arXiv 2410.24087](https://arxiv.org/abs/2410.24087) | Few-shot via in-context examples at inference — supervised-tuning-level accuracy, no training |
| Chronos-2 | Amazon | 2025 | [arXiv 2510.15821](https://arxiv.org/abs/2510.15821) | Univariate → universal: multivariate + covariates in-context; CRPS frontier |
| Moirai 2.0 | Salesforce | 2025 | [arXiv 2511.11698](https://arxiv.org/abs/2511.11698) | Decoder-only redesign; "less is more" — smaller, faster, stronger than Moirai 1 |

## Where it stands (2025-2026)
- **The frontier is multivariate + covariate-aware.** Chronos-2 and Toto fold cross-variate attention and known/past covariates in-context, closing the gap to task-specific models on real panels.
- **Benchmarks consolidated.** GIFT-Eval and BOOM are now the shared scoreboard; current leaders (TimesFM-2.5, Chronos-2, Toto, TiRex) trade the top spots — TimesFM-2.5 strong on MASE, Chronos-2 on CRPS.
- **Architecture is diversifying.** Decoder-only Transformers still dominate, but xLSTM (TiRex), flow matching (Sundial), and efficient long-conv/linear-RNN hybrids match far larger models — scale is not the only lever.
- **Few-shot beats zero-shot.** In-context fine-tuning lets a frozen model adapt from a handful of examples, narrowing the long-standing gap to fully supervised baselines.

## Transferred vs. reinvented
**Transferred from language modeling**
- Decoder-only Transformer, autoregressive next-token training.
- Tokenize-and-reuse the LLM stack wholesale (Chronos on T5).
- In-context learning: adapt at inference from examples in the prompt.

**Reinvented for time series**
- Patching as the token; value-bin or continuous-value vocabularies, not words.
- Scale/frequency invariance: instance norm, frequency embeddings, any-variate attention.
- Probabilistic decoders (quantiles, flow/diffusion) — forecasts are distributions, not classes.
- Leakage-controlled benchmarking (GIFT-Eval, BOOM); naive pretraining overlaps with test sets.

## Open problems
- **Is there a real "GPT-3 moment"?** Zero-shot beats classical baselines, but heterogeneity (scale, regime shifts, exogenous shocks) keeps a single universal model from dominating every domain.
- **Covariates and causality.** Most gains are still endogenous; cleanly conditioning on known future covariates and structural drivers remains unsolved.
- **Non-stationarity and distribution shift.** Models extrapolate trends and rare events poorly; calibration drifts under regime change.
- **Evaluation leakage.** Web-scale corpora silently contaminate test sets; "zero-shot" claims need provably disjoint pretraining data.

## See also
- [Language Foundation Models](../01-language/README.md) — the Transformer + tokenization recipe being ported here.
- [Audio & Speech Foundation Models](../04-audio-and-speech/README.md) — sibling sequential domain with patch/quantization tokens.
- [Climate & Earth Foundation Models](../13-climate-and-earth/README.md) — large-scale spatiotemporal forecasting.
- [Concept Library: Bricks](../../../library/bricks/README.md) — patching, tokenization, conformal prediction, and scaling-law primitives.
