# Potential Outcomes (Neyman–Rubin)

**One-liner:** Define causal effects as contrasts of per-unit counterfactual outcomes Y(1),Y(0); only one is ever observed (the "fundamental problem of causal inference"), but ATE = E[Y(1)−Y(0)] is identified from observational data under SUTVA + ignorability + overlap.

## The definition

Each unit i has two potential outcomes Y_i(1), Y_i(0) — what would happen under treatment vs. control. We observe only Y_i = Y_i(T_i) = T_i·Y_i(1) + (1−T_i)·Y_i(0); the other is the missing counterfactual.

- Unit-level effect: τ_i = Y_i(1) − Y_i(0)  (never observed directly)
- **ATE** = E[Y(1) − Y(0)],  **ATT** = E[Y(1) − Y(0) | T=1],  **CATE** τ(x) = E[Y(1) − Y(0) | X=x]

## The three identification assumptions

1. **SUTVA** — no interference (unit i's outcome doesn't depend on others' treatment) and one version of treatment (consistency: Y = Y(T)).
2. **Ignorability / unconfoundedness** — {Y(1),Y(0)} ⊥ T | X. Conditional on observed covariates, treatment is "as good as random." Untestable.
3. **Overlap / positivity** — 0 < P(T=1 | X=x) < 1 for all x. Every covariate profile could plausibly receive either arm.

Under these: E[Y(1) − Y(0)] = E_X[ E[Y | T=1, X] − E[Y | T=0, X] ]  (adjustment formula / g-formula).

## Estimators

- **IPW**: E[ TY/e(X) − (1−T)Y/(1−e(X)) ], with propensity e(X)=P(T=1|X).
- **AIPW / doubly robust**: consistent if *either* outcome model μ_t(X) or e(X) is correct.
- **DML / TMLE** (Chernozhukov 2018): cross-fit ML nuisances (μ, e), get √n-rate, valid CIs for ATE despite flexible learners.

## Where it appears

- **CATE meta-learners** — T/S/X/R/DR-learners; causal forests (Wager–Athey); EconML, DoWhy. Heterogeneous effect estimation for personalization/uplift.
- **A/B testing & off-policy eval** — IPW/doubly-robust estimators in RecSys and bandits are PO with the policy as treatment.
- **LLM/ML causal eval (2024–26)** — causal probing, "what if this token/feature were changed" framed as potential outcomes; PPI-style inference on ML-imputed counterfactuals.

## Common mistake

Treating ignorability as testable or guaranteed by "controlling for covariates." It is an assumption about *unobserved* confounders — no amount of data verifies it, and adjusting for a collider or post-treatment variable actively introduces bias. Also: PO answers interventional/counterfactual questions, not structural mechanism — it presumes a well-defined intervention exists.

## See also
- [[do-calculus]] — the SCM/graph view; do(T) interventions correspond to PO contrasts under consistency
- [[ppi]] — valid inference when outcomes are ML-predicted, same missing-data structure
- [[conformal]] — distribution-free uncertainty that composes with overlap/positivity concerns
