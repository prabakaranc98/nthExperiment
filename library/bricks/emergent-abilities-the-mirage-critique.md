# Emergent Abilities & the Mirage Critique

**One-liner:** Sharp, seemingly-discontinuous capability jumps with scale — argued by Schaeffer et al. (2023) to be largely artifacts of nonlinear/discontinuous metrics (e.g. exact-match accuracy) compressing a smooth underlying improvement, not genuine phase transitions.

## The key insight

"Emergence" (Wei et al. 2022, BIG-Bench) = a capability absent in small models that appears sharply past a scale threshold, when plotting a *discontinuous* metric vs scale. The mirage argument: per-token cross-entropy improves *smoothly* as a power law. A task needing all L tokens correct under exact-match gives

  Acc(N) ≈ p(N)^L,  where per-token correctness p(N) → 1 smoothly.

Raising a smooth curve to a high power L (or thresholding it) manufactures an apparent S-curve / cliff. Swap to a smooth, continuous metric — token edit distance, Brier score, log-likelihood of the answer — and the "emergence" usually vanishes into a gradual trend. Schaeffer et al. showed (a) emergence concentrates in nonlinear/discontinuous metrics (~92% in BIG-Bench), (b) you can *induce* fake emergence on vision models with a contrived metric, and (c) better resolution (more model checkpoints, larger eval sets) softens the cliffs.

## Where it appears

- Wei et al. 2022 "Emergent Abilities of LLMs" — the original claim; arithmetic, multi-step reasoning, instruction following appearing abruptly at GPT-3/PaLM scale.
- Schaeffer et al. 2023 "Are Emergent Abilities a Mirage?" (NeurIPS, Outstanding Paper) — the metric-artifact rebuttal.
- BIG-Bench / GPT-4 / PaLM evals — task selection now favors smooth, partial-credit metrics to forecast capability.
- Scaling-law forecasting — motivates predicting downstream perf from loss, not from brittle accuracy thresholds.

## Common mistake

Reading the mirage critique as "emergence doesn't exist / scaling is fully predictable." It shows many *reported* emergences are metric artifacts — it does not prove downstream capability is always smoothly predictable from loss. Genuinely hard-to-forecast, sharp transitions (and grokking-style dynamics) remain; the lesson is to distrust the metric before claiming a phase transition.

## See also
- [[scaling-laws]] — the smooth power-law in loss that emergence claims sit on top of
- [[scaling-law-extrapolation-brittleness]] — why predicting downstream behavior from loss is itself fragile
- [[grokking]] — a real sharp capability transition (in training time, not scale)
