# Deep Ensembles & MC-Dropout

**One-liner:** Approximate the Bayesian predictive distribution by averaging M independently-trained networks (deep ensembles) or M dropout-masked forward passes at test time (MC-dropout); deep ensembles remain the strongest, hardest-to-beat practical UQ baseline.

## The formula

Both approximate the posterior predictive by Monte Carlo over a set of weight samples {θ₁,...,θ_M}:

p(y|x) ≈ (1/M) Σ_m p(y|x, θ_m)

- **Deep ensembles** (Lakshminarayanan et al., 2017): θ_m = M full trainings from different random init (different data order/augmentation help too). Diversity comes from non-convex loss landscape → different modes.
- **MC-dropout** (Gal & Ghahramani, 2016): keep dropout ON at inference; each forward pass samples a Bernoulli mask, giving θ_m = θ ⊙ mask_m. Interpreted as variational inference with a Bernoulli q(θ).

**Total predictive uncertainty** decomposes (entropy / law of total variance):
H[ȳ] = E_θ H[p(y|x,θ)]  +  I[y;θ]   ( = aleatoric + epistemic )
The MC variance across members estimates the epistemic term; ensembles capture multi-modal disagreement, MC-dropout only local mode width.

## Where it appears

- **Lakshminarayanan et al. (2017)** — deep ensembles + adversarial training, beats Bayesian NNs on NLL/calibration; still the reference baseline in 2024-26 UQ papers.
- **Gal & Ghahramani (2016)** — MC-dropout as cheap "free" Bayesian inference for any net already using dropout.
- **Hyperparameter ensembles / BatchEnsemble / snapshot ensembles** — cheaper ensemble surrogates (shared backbone, rank-1 perturbations, cyclic-LR snapshots).
- **LLM UQ** — sampling multiple completions (a poor-man's ensemble) feeds semantic-entropy and self-consistency hallucination detectors; full LLM ensembles are usually too expensive, so dropout/sampling proxies dominate.

## Common mistake

Conflating ensemble *disagreement* with calibrated *probability*. Both methods improve calibration vs a single net but are not exact posteriors; MC-dropout in particular underestimates epistemic uncertainty (variational mode-collapse) and its quality is highly sensitive to dropout rate. Also: deep-ensemble accuracy gains come from averaging diverse modes, NOT from dropout-style regularization — they are different mechanisms.

## See also
- [[epistemic-vs-aleatoric-uncertainty]] — the decomposition these methods estimate via member variance
- [[calibration]] — the property both methods improve and are evaluated on
- [[weight-averaging-ema-model-soups]] — averaging weights (one model) vs averaging predictions (ensemble)
