# Calibration & ECE

**One-liner:** A model is calibrated when its predicted confidence equals empirical accuracy (of all predictions at 80% confidence, 80% are correct); Expected Calibration Error bins predictions by confidence and averages the gap; temperature scaling is the cheap post-hoc fix.

## The definition

Perfect calibration: for a classifier outputting confidence p̂ = max softmax,

P(ŷ = y | p̂ = p) = p   for all p ∈ [0,1]

## ECE (the binning estimator)

Partition predictions into M bins B₁..Bₘ by confidence. Then:

ECE = Σₘ (|Bₘ|/n) · |acc(Bₘ) − conf(Bₘ)|

where acc(Bₘ) = fraction correct in bin, conf(Bₘ) = mean confidence in bin. Plot acc vs conf per bin → **reliability diagram**; the gap to the diagonal is what ECE summarizes.

## Temperature scaling (the fix)

Fit a single scalar T > 0 on a held-out set, dividing logits before softmax:

softmax(z / T)

T > 1 softens (fixes overconfidence, the usual case); T < 1 sharpens. Optimize T by minimizing NLL on validation logits. It is monotonic, so it never changes argmax / accuracy — only the confidences.

## Where it appears

- **Guo et al. 2017** ("On Calibration of Modern Neural Networks") — showed deep nets are systematically overconfident vs. shallow ones; introduced temperature scaling as the strong baseline still used today.
- **LLM RLHF/DPO** — base-model log-probs are fairly calibrated, but RLHF degrades calibration (overconfident verbalized "95%"); a known alignment-tax symptom.
- **Selective prediction / abstention** — calibrated confidence is the gate for "answer vs. defer to human"; safety-evals report ECE alongside accuracy.
- **Ensembles & MC-dropout** — averaging improves calibration without a separate fit.

## Common mistake

Treating low ECE as proof of good per-instance uncertainty. ECE is a marginal, bin-averaged metric: a model can have ECE ≈ 0 while being badly wrong on every individual case (output the base rate for everyone). It is also sensitive to bin count/scheme and only scores the top-1 confidence, ignoring the rest of the distribution. Use proper scoring rules (NLL, Brier) to penalize the full predictive distribution.

## See also
- [[conformal]] — distribution-free calibrated *sets* with a coverage guarantee, vs. calibrated *probabilities*
- [[softmax]] — temperature scaling acts directly on the softmax temperature
- [[cross-entropy]] — NLL/log-loss is the proper scoring rule that ECE complements
