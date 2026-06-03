# Data Mixture / Domain Weighting (DoReMi, RegMix)

**One-liner:** Set the pretraining sampling proportions across domains (web, code, math, books, multilingual) via proxy-model methods — Group DRO (DoReMi), gradient alignment (DOGE), or regression over mixtures (RegMix) — to minimize loss / maximize downstream accuracy under a fixed token budget.

## The setup

You train on a mixture of k domains with weights α ∈ Δ^{k-1} (simplex, Σαᵢ = 1). αᵢ is the probability of drawing a batch from domain i. The question: what α minimizes target loss? Brute-force grid search over α is infeasible — proxy methods estimate it cheaply.

## DoReMi (Xie et al., 2023) — Group DRO

Train a small *reference* model at uniform/baseline weights. Then train a small *proxy* model that minimizes the worst-case *excess loss* over domains via online Group DRO, which updates per-domain weights toward domains with high excess loss:

  αᵢ ← αᵢ · exp(η · excessᵢ),  excessᵢ = ℓ_proxy(domain i) − ℓ_ref(domain i)

Renormalize α each step. The *time-averaged* α over training is the output mixture, then used to train the large model. Targets robustness, not a specific eval.

## RegMix (Liu et al., 2024) — regression

Train many tiny models (e.g. 1M params, 1B tokens) on random mixtures α⁽ʲ⁾. Fit a regressor (LightGBM) predicting loss from α. Then *simulate* over the simplex and pick the α minimizing predicted target loss. Cheaper to extrapolate; no reference model needed.

## DOGE (Fan et al., 2024) — gradient alignment

Upweight domains whose gradients align with the target-domain gradient (⟨gᵢ, g_target⟩), bilevel-style. Directly optimizes for a downstream target rather than worst-case.

## Where it appears

- **DoReMi** — used by PaLM-family / Gemini-era data pipelines; the proxy-then-scale recipe is the canonical baseline for principled mixing.
- **RegMix / regression mixing** — Llama-3, DCLM, and many 2024-25 open recipes use small-scale fit + extrapolation to set web/code/math ratios.
- **Data Mixing Laws (Ye et al., 2024)** — fits an explicit functional form L(α) and combines with scaling laws to predict the optimal mixture at the target scale.

## Common mistake

Assuming the proxy-optimal mixture transfers across scale and budget. Optimal α depends on model size, total token budget (repetition pressure under data-constrained regimes shifts it), and which downstream tasks you weight. A mixture tuned with a 1M-param proxy can be wrong at 70B — validate the extrapolation, don't blindly scale α.

## See also
- [[data-mixing-laws]] — fits L(α) functional forms to predict the optimal mixture at scale
- [[data-constrained-scaling-repetition-laws]] — repetition value shifts optimal domain weights under finite data
- [[pretraining-data-curriculum]] — ordering/phasing data, the temporal complement to static mixture weights
