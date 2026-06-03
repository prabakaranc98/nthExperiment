# Conformal Prediction

**One-liner:** Distribution-free, finite-sample prediction sets with a guaranteed coverage probability — valid under exchangeability only, no assumptions about the model or data distribution.

## The guarantee

For any model f, calibration set {(x₁,y₁), ..., (xₙ,yₙ)}, and test point (xₙ₊₁, yₙ₊₁) drawn exchangeably:

P(yₙ₊₁ ∈ Ĉ(xₙ₊₁)) ≥ 1−α

where Ĉ is a prediction set constructed from the calibration nonconformity scores.

**No assumptions about the model.** Works with any black-box predictor.

## Split conformal (the practical version)

1. Choose a **nonconformity score** s(x, y) — higher means "x and y don't fit well" (e.g., 1 − softmax(y|x) for classification)
2. Compute scores on calibration set: s₁, ..., sₙ
3. Find the ⌈(n+1)(1−α)⌉/n quantile q̂ of the calibration scores
4. Prediction set: Ĉ(x) = {y : s(x,y) ≤ q̂}

## Where it appears

- **LLM evaluation** — Conformal Language Modeling (Quach et al., ICLR 2024): output sets with coverage guarantees on generated text
- **Prediction-Powered Inference** — PPI uses conformal ideas for valid inference on ML-labeled data
- **Calibration** — conformal gives calibration *by construction* rather than post-hoc calibration
- **Safety-critical deployment** — any time you need guaranteed error rates (medical, legal, financial)

## The exchangeability requirement

Conformal is valid under **exchangeability** (weaker than i.i.d.) — the joint distribution of calibration + test points is invariant to permutation. This fails under:
- Distribution shift (test ≠ calibration)
- Time series (temporal dependence)

**Fix:** weighted conformal prediction (Tibshirani et al., 2019) reweights calibration scores by importance weights to handle covariate shift.

## Common mistake

Thinking conformal gives you calibrated *probabilities*. It gives you calibrated *sets* — you know the set contains the true label with probability ≥ 1−α, but not the probabilities of individual elements within the set.

## See also
- [[ppi]] — Prediction-Powered Inference; uses conformal ideas for inference
- [[kl-divergence]] — proper scoring rules (log-loss) vs. conformal coverage guarantee
