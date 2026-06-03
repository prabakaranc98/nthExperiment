# Prediction-Powered Inference (PPI)

**One-liner:** Estimate a population quantity (mean, regression coefficient, quantile) using a large pool of ML predictions, then debias with a small gold-labeled set — yielding valid, tighter confidence intervals than gold-only inference.

## The estimator

Goal: estimate θ* (e.g., E[Y]). You have n labeled pairs (Xᵢ, Yᵢ) and N ≫ n unlabeled Xⱼ, plus a black-box predictor f. Naive use of f(X) is biased; gold-only ignores N. PPI corrects the prediction-based estimate by the measured bias ("rectifier") on the labeled set:

θ̂_PPI = (1/N) Σⱼ f(Xⱼ)  −  (1/n) Σᵢ [ f(Xᵢ) − Yᵢ ]
        └── cheap, low-variance ─┘   └── bias correction (rectifier) ─┘

For the mean: the first term is the prediction average over all N; the second is the average prediction error on gold data. The two f-terms cancel in expectation, so θ̂_PPI is **unbiased regardless of how good f is** (consistent for any f). Generalizes to M-estimators (means, quantiles, GLM coefficients) via debiasing the estimating equation.

## PPI++ (the version you should use)

PPI++ (Angelopoulos et al., 2023) adds a tuning weight λ ∈ ℝ on the rectifier:

θ̂_λ = (1/N) Σⱼ λ·f(Xⱼ)  −  (1/n) Σᵢ [ λ·f(Xᵢ) − Yᵢ ]

Optimal λ* trades off how much to trust f. λ=0 recovers the classical gold-only estimator (so PPI++ is **never worse** than gold-only, asymptotically); λ=1 is vanilla PPI. CI half-width shrinks roughly with the gold-prediction correlation: effective sample size ≈ n / (1 − ρ²).

## Where it appears

- **Angelopoulos, Bates, Jordan et al. (2023, Science)** — original PPI for means/quantiles/GLMs; PPI++ adds λ tuning and power tuning.
- **LLM-as-judge / autorater eval** — use cheap LLM labels on a big unlabeled set + a few hundred human labels to get valid CIs on model win-rates, accuracy, toxicity rates without massive human annotation.
- **Computational social science / genomics** — AlphaFold structures, ML-imputed phenotypes used as outcomes with rigorous downstream inference (Stratified/Prediction-Powered GWAS).
- **Active/Stratified PPI** — choose *which* points to label to maximize CI tightening.

## Common mistake

Thinking PPI needs an accurate model to be valid. It does **not** — validity (coverage, unbiasedness) holds for *any* f, even a terrible one; f only affects *efficiency* (interval width). The flip side: if f is uncorrelated with Y, PPI gives no width reduction over gold-only (and naive PPI can be slightly *worse* — use PPI++ with λ tuning to avoid this). Also requires the labeled set to be a random/exchangeable sample from the same population as the unlabeled pool.

## See also
- [[conformal]] — both give distribution-free, model-agnostic validity from a small calibration set; conformal does prediction sets, PPI does parameter inference
- [[bias-variance]] — PPI is exactly a control-variate bias-variance trade: kill bias with gold, cut variance with predictions
- [[calibration]] — alternative way to make ML outputs statistically trustworthy, but PPI targets population estimands, not per-point probabilities
