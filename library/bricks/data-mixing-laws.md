# Data Mixing Laws

**One-liner:** Predictive scaling laws expressing validation loss as a function of domain mixture proportions (plus N and D), fit on cheap small runs to extrapolate and optimize the mixture for the large target run *before* spending the compute.

## The formula (Ye et al., 2024)

For K domains with mixture weights r = (r₁,...,r_K), ∑r_i = 1, the loss on a target domain (or weighted total) is modeled as an **exponential of a linear combination** of the proportions:

L(r) = c + k · exp( ∑ᵢ tᵢ rᵢ )

i.e. log(L − c) is *linear* in the mixture weights. Fit constants {c, k, tᵢ} on a small grid of mixtures at small (N, D), then predict L(r) for any unseen mixture and minimize over the simplex to pick r*.

**Nested / functional view:** chain with Chinchilla. First fit L vs (N,D) per mixture, then fit how those coefficients move with r — letting you extrapolate the *mixing law fit at small scale* to the target N, D. Optimize r* once on the predicted large-scale surface.

## Where it appears

- **Data Mixing Laws (Ye et al., 2024)** — the exponential-linear law; predicts the loss of an unseen RedPajama mixture and finds a mixture beating the default with the same compute.
- **DoReMi (Xie et al., 2023)** — learns domain weights via group-DRO on a small proxy model, then trains the big model with those fixed weights; a learned-not-fit cousin.
- **RegMix (Liu et al., 2024)** — trains many tiny models, fits a regression (mixing law) predicting loss from proportions, extrapolates the optimum.
- **Frontier pretraining** — choosing web/code/math/multilingual ratios; annealing-phase upweighting of high-quality data uses the same predict-then-set logic.

## Common mistake

Assuming the fitted optimal mixture transfers across scale and objective. The coefficients drift with N, D, tokenizer, and dedup; and minimizing *total* loss ≠ minimizing loss on the domain you care about (web dominates the average). Optimize the target-weighted loss, and validate that the small-scale fit actually extrapolates — it is brittle near the simplex edges and under repetition.

## See also
- [[data-mixture-domain-weighting]] — the decision the law is optimizing
- [[scaling-laws]] — the (N,D) backbone these laws extend with a mixture axis
- [[data-constrained-scaling-repetition-laws]] — repetition interacts with per-domain token budgets
