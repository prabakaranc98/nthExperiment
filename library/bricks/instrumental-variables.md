# Instrumental Variables (IV)

**One-liner:** When ignorability fails (unobserved confounder U affects both treatment D and outcome Y), recover the causal effect using an instrument Z that affects D but reaches Y *only through* D — so variation in D induced by Z is "as good as random."

## The three conditions

For instrument Z, treatment D, outcome Y, confounder U:

1. **Relevance:** Cov(Z, D) ≠ 0 — Z actually moves D.
2. **Exclusion:** Z affects Y only through D (no direct Z→Y arrow, no Z→U).
3. **Exogeneity / independence:** Z ⊥ U (Z is unconfounded with the error).

Linear model Y = βD + U, D = πZ + V. Then β is identified by the ratio of reduced-form to first-stage:

  β = Cov(Z, Y) / Cov(Z, D)

**2SLS:** (1) regress D on Z to get fitted D̂ = projection of D onto Z; (2) regress Y on D̂. Stacks to β̂ = (Z'D)⁻¹Z'Y in the just-identified case; with covariates X, project out X first.

**Heterogeneous effects (LATE):** with a binary instrument, IV recovers not the ATE but the **Local Average Treatment Effect** — the effect for *compliers* (units whose D flips with Z), under a monotonicity (no-defiers) assumption (Imbens & Angrist).

## Where it appears

- **2SLS / classic econometrics** — Mendelian randomization (genotype as Z), randomized-encouragement designs, judge/quota instruments.
- **Deep IV (Hartford et al., ICML 2017)** — two neural nets estimate P(D|Z,X) and the structural function; replaces linear 2SLS for nonlinear, high-dim settings.
- **DML / orthogonal IV** — double/debiased ML uses IV moment conditions with cross-fitting for valid inference under flexible nuisance estimators.
- **RecSys & LLM eval** — natural experiments / encouragement as instruments to debias logged feedback; IV framing for off-policy effects.

## Common mistake

Using a **weak instrument** (small Cov(Z,D), low first-stage F). Weak instruments make 2SLS badly biased *toward OLS* and inflate variance — and you cannot test exclusion (condition 2) from data; it is an untestable assumption you must defend. Rule of thumb: first-stage F ≫ 10 (modern work demands much higher).

## See also
- [[potential-outcomes]] — IV identifies the LATE, a compliers-only contrast of Y(1)−Y(0)
- [[backdoor-criterion-confounding-adjustment]] — the alternative when confounders *are* observed; IV is for when they are not
- [[double-debiased-machine-learning]] — orthogonal moment conditions and cross-fitting for IV with ML nuisances
