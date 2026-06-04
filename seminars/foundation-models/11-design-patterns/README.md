# 11 · Design Patterns & Theory of Foundation Models

*The meta-domain: the recurring recipe that makes pretrain-then-adapt transfer across every modality.*

## The adaptation

- **Token** — there is no fixed token. Each domain invents one: subword, patch, amino acid, codon, time-step, atom, mesh node, action. The recipe is "find a discrete or continuous unit dense in self-supervised signal."
- **Objective** — masked, next-step, denoising, or contrastive prediction. All recover removed information from context; the bet is that solving this forces world structure into the weights.
- **Inductive bias** — minimal but load-bearing: causal masking for generation, equivariance for physics/structure, patchification for spatial data. Scale substitutes for bias only when data is abundant.
- **Verification** — the signal that survives transfer: held-out labels, simulation, a test suite, a fold, a forecast that comes true. Domains without it (no ground truth) resist the recipe.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| Scaling Laws for Neural Language Models | OpenAI | 2020 | [arXiv 2001.08361](https://arxiv.org/abs/2001.08361) | Loss is a smooth power law in compute, data, params — scaling becomes predictable, not magic |
| On the Opportunities and Risks of Foundation Models | Stanford CRFM | 2021 | [arXiv 2108.07258](https://arxiv.org/abs/2108.07258) | Named the paradigm; framed emergence + homogenization as the two defining properties |
| A Path Towards Autonomous Machine Intelligence (JEPA) | LeCun / Meta | 2022 | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | Predict in latent space, not pixels — the dominant non-generative blueprint for vision/video/world models |
| Training Compute-Optimal LLMs (Chinchilla) | DeepMind | 2022 | [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) | ~20 tokens/param; reset the compute-allocation default for every domain that followed |
| Scaling Data-Constrained Language Models | Muennighoff et al. | 2023 | [arXiv 2305.16264](https://arxiv.org/abs/2305.16264) | Up to ~4 epochs of repeated data ≈ fresh data — the key result once a modality runs out of tokens |
| Are Emergent Abilities a Mirage? | Stanford | 2023 | [arXiv 2304.15004](https://arxiv.org/abs/2304.15004) | Many "sharp" emergences are metric artifacts — recalibrated how we read capability jumps |
| The Platonic Representation Hypothesis | MIT | 2024 | [arXiv 2405.07987](https://arxiv.org/abs/2405.07987) | Large models across vision and language converge to a shared geometry — the why behind cross-domain transfer |
| Scaling LLM Test-Time Compute Optimally | DeepMind / UC Berkeley | 2024 | [arXiv 2408.03314](https://arxiv.org/abs/2408.03314) | Spending compute at inference can beat a larger model — opened the second scaling axis |
| An Empirical Study of Scaling Laws for Transfer | — | 2024 | [arXiv 2408.16947](https://arxiv.org/abs/2408.16947) | A measurable "transfer gap" term predicts when pretraining on A helps task B — the quantitative core of this page |

## Where it stands (2025-2026)

- **Two scaling axes, not one.** Train-time scaling has been joined by test-time compute as a first-class lever; the open question is the optimal split of a fixed budget between them.
- **Beyond Chinchilla's 20:1.** Inference cost now dominates lifetime compute, so practice has swung hard toward overtraining small models — production ratios reach tens of thousands of tokens/param (Qwen3, Liquid LFM2.5).
- **Scaling laws went domain-specific.** "Scaling" is now a family of regime-dependent regularities, not one universal curve; cross-domain transfer itself follows predictable, asymmetric power laws (e.g. 3D medical imaging, network biology).
- **Convergence is measurable.** Representation-alignment evidence (Platonic Hypothesis and follow-ups) reframes transfer as models discovering a shared statistical model of reality — giving a mechanistic story for why one recipe ports across modalities.

## Transferred vs. reinvented

**Transferred directly from language modeling**
- The pretrain → adapt → align pipeline and the Transformer backbone.
- Self-supervised next-/masked-token objectives over a domain-specific vocabulary.
- Power-law scaling intuition and compute-optimal allocation.

**Reinvented per domain**
- Tokenization (patches, residues, atoms, actions) — the hardest, least transferable design choice.
- Inductive biases: equivariance, conservation laws, spatial/temporal structure.
- The verification signal and its metric — what counts as "correct" is domain-native.

## Open problems

- **A predictive theory of transfer.** The transfer-gap law is empirical and post-hoc; we still cannot forecast *a priori* which source domain will help a target.
- **Sharp capability jumps.** Even after the mirage critique, genuine phase transitions in reasoning and tool use remain hard to anticipate or schedule.
- **Recipe limits.** Data-scarce, non-compositional, or unverifiable domains (scarce ground truth) resist the paradigm — no amount of scale substitutes for a missing self-supervised signal.

## See also

- [01 · Language Foundation Models](../01-language/README.md) — where every pattern here was first established
- [02 · Vision Foundation Models](../02-vision/README.md) — patches, JEPA, and the masked-prediction transplant
- [10 · Multimodal Foundation Models](../10-multimodal/README.md) — convergence and the Platonic Hypothesis in practice
- [Concept Library — Bricks](../../../library/bricks/README.md) — scaling laws, conformal prediction, and reusable components
