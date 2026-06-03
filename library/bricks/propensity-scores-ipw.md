# Propensity Scores & IPW

**One-liner:** The propensity score e(x)=P(T=1∣X=x) is the coarsest balancing score — conditioning on it suffices to remove confounding under ignorability + overlap; inverse-probability weighting (IPW) reweights units by 1/P(observed treatment) to synthesize a pseudo-population where treatment is independent of X.

## The definition

Propensity score: e(X) = P(T=1 ∣ X), estimated by logistic regression or a flexible ML model.

**Balancing property** (Rosenbaum & Rubin, 1983): X ⊥ T ∣ e(X). So if ignorability holds given X, it also holds given the scalar e(X) — you can adjust on one number instead of all confounders.

**IPW (Horvitz–Thompson) estimator of the ATE:**

ÂTE = (1/n) Σᵢ [ Tᵢ Yᵢ / ê(Xᵢ) − (1−Tᵢ) Yᵢ / (1−ê(Xᵢ)) ]

A treated unit with low e(X) (rare to be treated) gets a large weight, standing in for similar untreated units. The stabilized / Hájek variant normalizes by Σ Tᵢ/ê instead of n to reduce variance.

## Where it appears

- **Doubly-robust / AIPW estimators** — combine an IPW term with an outcome model; consistent if *either* the propensity model *or* the outcome model is right. The basis for [[double-debiased-machine-learning]] (DML), which uses cross-fitting + Neyman-orthogonal scores.
- **Off-policy RL & bandits** — IPS/IPW reweighting by π_target/π_behavior is exactly the [[importance-sampling-the-off-policy-ratio]] correction; counterfactual policy evaluation and logged-feedback recsys.
- **LLM eval & survey reweighting** — propensity weights to correct sampling/selection bias when audiences or prompts are non-random.

## Common mistake

Ignoring **overlap (positivity)**. If ê(X)→0 or →1 for some strata, weights explode and the estimator has huge variance and bias — a handful of units dominate. Always check the propensity distribution; trim or clip extreme weights. IPW also requires a *correctly specified* treatment model — a misspecified e(X) gives a biased ATE (which is why DR/AIPW is preferred).

## See also
- [[potential-outcomes]] — the ignorability + overlap assumptions that make IPW identify the ATE
- [[backdoor-criterion-confounding-adjustment]] — graphical justification for which X to put in e(X)
- [[importance-sampling-the-off-policy-ratio]] — IPW is the same reweighting trick in the RL/off-policy setting
