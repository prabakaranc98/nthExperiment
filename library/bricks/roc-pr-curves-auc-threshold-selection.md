# ROC/PR Curves, AUC & Threshold Selection

**One-liner:** Threshold-free summaries of a binary scorer's ranking quality — ROC plots TPR vs FPR, PR plots precision vs recall; AUC is the area under each; the deployed threshold is a separate decision driven by the operating cost/base-rate, not by the AUC.

## The definitions

For a score s(x) and threshold t, predict positive if s(x) ≥ t. Sweep t over all values:

- TPR = recall = TP/(TP+FN)   FPR = FP/(FP+TN)
- Precision = TP/(TP+FP)

**ROC** = curve of (FPR, TPR) as t varies. **PR** = curve of (recall, precision).

**ROC-AUC** equals the probability a random positive scores above a random negative (= the Mann–Whitney U / Wilcoxon statistic, a pure ranking metric): AUC = P(s(x⁺) > s(x⁻)). Chance = 0.5.

**PR-AUC** (average precision) ≈ Σ (Rₙ − Rₙ₋₁)·Pₙ. Baseline = the positive prevalence π, not 0.5.

## Threshold selection

AUC summarizes all thresholds; deployment picks **one**. Choose by the operating point you actually care about:
- Fix FPR (e.g. ≤ 1% false-flag budget for a safety filter), read off the achievable TPR.
- Maximize F_β = (1+β²)·P·R / (β²·P + R) for a chosen recall/precision tradeoff β.
- Cost-optimal: t* where slope = (cost_FP / cost_FN)·((1−π)/π) on the ROC.
Then calibrate scores (Platt/isotonic/temperature) if you need the threshold to mean a probability.

## Where it appears

- **Safety / moderation classifiers** (Llama Guard, OpenAI moderation, jailbreak detectors) — reported at fixed low FPR; the shipped threshold trades catch-rate vs over-refusal.
- **Hallucination / uncertainty detection** (semantic entropy, p(true), SelfCheckGPT) — AUROC is the standard metric for "does the score separate correct from hallucinated answers."
- **Retrieval / reranking** — PR curves and AP/MAP-style metrics over relevance; PR preferred because relevant docs are rare.
- **Membership-inference / extraction attacks** — measured by TPR at very low FPR (high-precision regime), where AUC alone is misleading.

## Common mistake

Using ROC-AUC on heavily imbalanced problems. With rare positives, a flood of true negatives keeps FPR tiny, so ROC-AUC stays optimistically high while the model is useless at the operating point. Use **PR-AUC** (or TPR@fixed-low-FPR) when positives are rare — and remember PR's baseline is prevalence π, so an AP must be compared against π, not 0.5.

## See also
- [[calibration]] — turning a thresholdable score into a trustworthy probability
- [[semantic-entropy-for-hallucination-detection]] — a detector scored almost universally by AUROC
- [[bootstrap-confidence-intervals-for-eval-metrics]] — how to put error bars on an AUC
