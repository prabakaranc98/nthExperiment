# Membership Inference & Training-Data Extraction

**One-liner:** Membership inference (MIA) tests whether a given sample was in the training set; extraction attacks make a model regurgitate verbatim memorized data — together the empirical ground truth behind privacy, memorization, and "did it train on my data" claims.

## The attacks (two related primitives)

**MIA** — given target sample x and model M, output 1 if x ∈ train. The signal is that training points have anomalously low loss / high confidence. The strong modern form is the **Likelihood Ratio Attack (LiRA, Carlini et al. 2022)**: train many shadow models with/without x, fit Gaussians to the (logit-scaled) confidence under each, and threshold the likelihood ratio:

  Λ(x) = log [ p(conf(x) | x ∈ train) / p(conf(x) | x ∉ train) ]

LiRA is *per-example* and calibrated to each sample's intrinsic hardness — a loss-threshold attack is the crude special case.

**Extraction** — sample/greedy-decode from M and check generations against the training corpus. A string is **k-eidetic memorized** if it is emittable yet appears ≤ k times in train. Verify with membership (high-confidence completion) + corpus search.

## The evaluation rule

Report attack power as **TPR at low FPR** (e.g. TPR@FPR=0.1%) on a log-log ROC, *not* balanced accuracy or AUC. Privacy is about the few highly-exposed outliers, so average-case accuracy near 50% can hide a catastrophic worst-case leak.

## Where it appears

- **Carlini et al. 2021 ("Extracting Training Data from GPT-2")** — black-box generation + membership ranking recovers verbatim PII, code, URLs; memorization scales with model size, duplication, and context length.
- **LiRA / "Membership Inference Attacks From First Principles" (2022)** — the per-example shadow-model attack that reset the SOTA bar and the TPR@low-FPR metric.
- **DP-SGD audits & "privacy auditing"** — MIA used as the *empirical lower bound* on the ε that a DP claim must beat.
- **Copyright / data-provenance litigation & "scalable extraction" (2023, divergence attack on aligned chat models)** — evidence that production LLMs emit training text.

## Common mistake

Reporting AUC or balanced accuracy and concluding "no leakage" because it's ~50%. The threat is the tail: a handful of points at TPR@0.1%FPR ≫ baseline is a real breach. Also: extraction proves memorization but a *failed* attack proves nothing — it is a lower bound on leakage, never a privacy guarantee.

## See also
- [[differential-privacy]] — the formal defense; MIA is how you audit whether the promised ε actually holds
- [[memorization-vs-generalization]] — extraction is the empirical instrument that measures memorization
- [[machine-unlearning]] — MIA is the standard verifier that "forgotten" data is truly gone
