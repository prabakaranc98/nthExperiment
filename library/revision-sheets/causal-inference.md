# Causal Inference — Revision Sheet

One page from "what is a causal effect" to the modern ML estimators.

## Two languages (they agree)

| Potential Outcomes (Neyman–Rubin) | Structural / Graphical (Pearl) |
|-----------------------------------|--------------------------------|
| Each unit has Y(1), Y(0); only one is observed | A causal DAG; arrows = direct causes |
| Effect = contrast of counterfactuals | Effect = `P(Y | do(X))` vs `P(Y | X)` |
| Assumptions: SUTVA, ignorability, overlap | Assumptions read off the graph (d-separation) |

**The fundamental problem of causal inference:** you never see both Y(1) and Y(0) for the same unit. Causal inference is a missing-data problem.

## Estimands

```
ATE  = E[Y(1) − Y(0)]                    (average treatment effect)
ATT  = E[Y(1) − Y(0) | T=1]              (effect on the treated)
CATE = E[Y(1) − Y(0) | X=x]              (heterogeneous / conditional)
```

## Identification — when can a causal effect be estimated from data?

- **Ignorability / no unmeasured confounding:** `{Y(0),Y(1)} ⟂ T | X`. Then `ATE = E_X[ E[Y|T=1,X] − E[Y|T=0,X] ]`.
- **Backdoor criterion:** adjust for a set Z that blocks all back-door paths X←…→Y and contains no descendants of X. That Z is sufficient to deconfound.
- **Frontdoor criterion:** identify through a mediator when a confounder is unobserved.
- **Overlap (positivity):** `0 < P(T=1|X) < 1` — every covariate profile could have gotten either treatment.

## Estimators

| Method | Idea | Watch out |
|--------|------|-----------|
| **Regression adjustment** | model E[Y|T,X], plug in | misspecification bias |
| **IPW** (propensity weighting) | reweight by 1/P(T|X) | unstable with extreme weights |
| **Doubly robust / AIPW** | combine outcome + propensity models | consistent if *either* is right |
| **Matching** | pair treated/control on X or propensity | curse of dimensionality |
| **Instrumental variables (IV)** | use Z that affects Y only through T (2SLS) | weak/invalid instruments |
| **DiD / RDD** | exploit time or a threshold | parallel-trends / continuity assumptions |

## Double / Debiased ML (DML) — the modern workhorse

Estimate nuisances (outcome model, propensity) with **any** ML model, but use **Neyman-orthogonal** scores + **cross-fitting** (train nuisances on one fold, evaluate the effect on another). Result: √n-consistent, asymptotically normal effect estimates with valid CIs even when the ML nuisances converge slowly.

## Common mistake

"Controlling for more variables is safer." **No** — conditioning on a *collider* or a *mediator* opens spurious paths and *creates* bias. Draw the DAG; adjust for confounders, never for colliders/mediators.

## See also
- [[potential-outcomes]] · [[do-calculus]] · [[causal-representation-learning]] · [[ppi]]
