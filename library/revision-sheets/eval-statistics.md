# Evaluation Statistics — Revision Sheet

One page on putting **valid error bars** on model comparisons — the difference between a number and a result.

## The mindset

A benchmark score is a **sample statistic**, not a constant. "Model A scores 71.2, Model B 70.8" means nothing without a CI and a test. Treat every eval as an experiment.

## Confidence intervals on a score

```
Accuracy p over n items:  SE = sqrt( p(1−p) / n )
95% CI ≈ p ± 1.96·SE        (Wald; use Wilson for small n or p near 0/1)
```

- **Bootstrap** when the metric isn't a simple mean (F1, BLEU, win-rate, pass@k): resample items with replacement B times, recompute, take the 2.5/97.5 percentiles. The one most-skipped, most-needed step.

## Comparing two models — pair it

Models are evaluated on the **same items** → use a **paired** test, which removes item-difficulty variance and is far more powerful than unpaired.

| Data | Test |
|------|------|
| Paired correct/incorrect | **McNemar** (on the discordant pairs) |
| Paired continuous scores | **paired t** / Wilcoxon signed-rank |
| Win/lose/tie preferences | **sign test** / bootstrap on win-rate |

## Power & design (decide n *before* running)

```
MDE (minimum detectable effect) ↓  ⇒  n ↑   (n ∝ 1/MDE²)
Underpowered eval → "no significant difference" is uninformative, not reassuring.
```

- **Multiple comparisons:** testing many models/metrics inflates false positives → Bonferroni / Benjamini–Hochberg (FDR).
- **Sequential / peeking:** repeatedly checking a rolling eval inflates error → use always-valid p-values (e-values, confidence sequences).

## Cheap labels, valid inference — PPI

Have a few gold labels + many LLM-judge labels? **Prediction-Powered Inference** debiases the ML labels with the gold set → valid CIs that are *tighter* than using gold alone. The rigorous way to use LLM-as-judge at scale.

## Distribution-free guarantees — conformal

**Conformal prediction** gives sets with finite-sample coverage `P(y ∈ C(x)) ≥ 1−α` assuming only exchangeability (no model/distribution assumptions). Calibrate a nonconformity quantile on held-out data.

## Calibration (do the probabilities mean anything?)

- **ECE** bins predictions by confidence and measures |confidence − accuracy| per bin. RLHF often *degrades* calibration.
- **Temperature scaling**: divide logits by a single learned T — cheap post-hoc fix that doesn't change the argmax.

## Common mistake

Reporting a single number with no interval, no paired test, and no n. A 0.4-point gap on 500 items is almost certainly noise.

## See also
- [[conformal]] · [[ppi]] · [[calibration]] · [[kl-divergence]]
