# Double / Debiased Machine Learning (DML)

**One-liner:** Estimate a low-dimensional causal effect at √n rate by plugging flexible ML nuisance estimates into a Neyman-orthogonal moment and cross-fitting — orthogonality kills first-order nuisance error, cross-fitting kills overfitting bias.

## The recipe (Chernozhukov et al., 2018)

Target the partially linear model Y = θD + g₀(X) + ε, E[ε|X,D]=0, with treatment D = m₀(X) + V. Naively plugging in an ML estimate ĝ biases θ̂ (regularization/overfitting leaks into the effect). DML fixes this with two ingredients:

**1. Neyman orthogonality** — use a moment ψ whose Gateaux derivative w.r.t. the nuisance η vanishes at the truth: ∂_η E[ψ(W;θ₀,η₀)] = 0. For the PLR, this is the residual-on-residual (Robinson/FWL) moment:

ψ = (Y − ĝ(X)) − θ(D − m̂(X)) · (D − m̂(X))

so θ̂ = E[(D−m̂)(Y−ĝ)] / E[(D−m̂)²]. First-order errors in ĝ, m̂ enter only as their *product*.

**2. Cross-fitting (K-fold)** — fit nuisances on the complement of fold k, evaluate the moment on fold k, average. Breaks the dependence that would otherwise inflate bias.

Result: θ̂ is √n-consistent and asymptotically normal even when ĝ, m̂ converge at slow rates oₚ(n^{-1/4}) — because the bias is the *product* of the two nuisance errors.

## Where it appears

- **EconML / DoubleML** — Microsoft's libraries; DML, DR-learner, and the AIPW/orthogonal estimators for CATE
- **Causal forests (GRF, Athey-Wager)** — orthogonalized "R-learner" splitting uses residualized outcomes/treatments
- **Heterogeneous effects with LLMs (2024-26)** — text/embedding features as high-dim X; DML to estimate ATE while controlling for everything an ML model can read
- **AIPW / doubly-robust estimators** — the augmented IPW moment is the orthogonal score for the ATE (g and propensity m as the two nuisances)

## Common mistake

Skipping cross-fitting and reusing the same data to fit nuisances and evaluate the moment. Orthogonality alone is not enough — without sample splitting the overfit nuisance error correlates with the score, reintroducing first-order bias and invalidating the √n CLT. Also: orthogonality only protects the *targeted* parameter θ; the nuisances themselves are not √n-estimable.

## See also
- [[propensity-scores-ipw]] — the propensity m₀(X) is one of the two DML nuisances; AIPW is its orthogonal upgrade
- [[potential-outcomes]] — the estimand (ATE/CATE) DML targets, under unconfoundedness
- [[ppi]] — same flavor: use an ML model for power but keep valid √n inference on the target
