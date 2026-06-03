# Scaling-Law Extrapolation & Brittleness

**One-liner:** Fitting a power law on small/cheap runs to predict the loss of a much larger run — reliable only when the fitted exponent, irreducible-loss offset, and all confounders (architecture, tokenizer, data quality, optimizer, repetition) are held fixed, which they rarely are.

## The formula / definition

The Chinchilla-style fit (Hoffmann et al., 2022) is the workhorse:

  L(N, D) = E + A·N^{-α} + B·D^{-β}

- **E** = irreducible loss (entropy of the data / Bayes floor). Ignoring it and fitting a pure `L ∝ C^{-c}` line is the #1 extrapolation error: the unmodeled offset makes the slope you read off small runs steeper than the true asymptotic slope, so you over-predict the gains of scale.
- Fit `{E, A, B, α, β}` via Huber loss in log-space over a grid of small runs (IsoFLOP profiles), then extrapolate to the target compute.

**Broken neural scaling laws (BNSL, Caballero et al., 2023):** a single power law is often wrong. Reality is a smoothly-broken law — a product of power-law segments with breaks:

  L(x) = a + b·x^{-c₀} · Π_i [1 + (x/d_i)^{1/f_i}]^{-c_i·f_i}

i.e. the exponent *changes* at breakpoints (and can even bend non-monotonically — inflections, double descent, emergence). Extrapolating across an unseen break is the failure mode.

## Where it appears

- **Chinchilla & its replications** — the 2022 fit was later found sensitive to the optimizer/LR-decay setup; runs without proper cosine decay-to-end mis-estimate D-exponents, shifting compute-optimal N*/D*.
- **GPT-4 technical report** — predicted final loss/HumanEval from runs using ≤1/1000th the compute; the public proof-of-concept that *careful* extrapolation works when the recipe is frozen.
- **Data-constrained scaling (Muennighoff et al.)** & **data-mixing laws** — repeated tokens have a *decaying* effective value, breaking the clean D^{-β} term; mixture weights shift exponents.
- **muP / hyperparameter scaling laws** — extrapolation is only valid if HPs transfer; an un-transferred LR makes the large run land off the fitted curve.
- **Emergence debates** — apparent "breaks" on downstream metrics are often artifacts of discontinuous/thresholded scoring, not of the loss curve.

## Common mistake

Reading the exponent off a log-log line and assuming it is a universal constant. The slope is entangled with the irreducible offset E and with every held-fixed nuisance: change the tokenizer, data quality, depth/width ratio, optimizer, or repetition regime and the fit silently re-parameterizes. A power law that fits 6 small runs almost always interpolates well and tells you nothing about whether a break lies between you and your target scale.

## See also
- [[scaling-laws]] — the base power laws this card stress-tests
- [[data-constrained-scaling-repetition-laws]] — repetition is a concrete source of exponent shift / broken laws
- [[emergent-abilities-the-mirage-critique]] — metric-induced "breaks" vs. real ones in extrapolation
