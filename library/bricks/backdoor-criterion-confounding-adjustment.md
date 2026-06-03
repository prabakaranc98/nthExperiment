# Backdoor Criterion & Confounding Adjustment

**One-liner:** A graphical test for "what to control for" — a covariate set Z that blocks every back-door path from X to Y (and contains no descendant of X) licenses the adjustment formula P(Y|do(x)) = Σ_z P(Y|x,z)P(z), turning a confounded association into an unbiased causal effect.

## The definition

A **back-door path** from X to Y is any path with an arrow pointing *into* X (e.g. X ← Z → Y) — these carry spurious, non-causal association. Z satisfies the **backdoor criterion** for (X→Y) if:

1. No node in Z is a descendant of X, and
2. Z **blocks** every back-door path from X to Y (via d-separation).

A path is *blocked by Z* if it contains either a chain/fork node in Z (… → m → … or … ← m → …, m ∈ Z), or a **collider** node m (… → m ← …) such that neither m **nor any descendant of m** is in Z. Then:

  P(Y | do(x)) = Σ_z P(Y | x, z) P(z)   (**adjustment / g-formula**)

Backdoor adjustment is the workhorse case of the do-calculus ID algorithm; ignorability {Y(1),Y(0)} ⊥ T | Z in potential-outcomes is its counterfactual twin.

## Where it appears

- **DoWhy / EconML / Ananke** — `identify_effect` searches the DAG for a valid backdoor (or minimal/optimal) adjustment set before any estimation.
- **Double/debiased ML (Chernozhukov)** — estimates the backdoor adjustment with cross-fit ML nuisance models μ(X,Z), e(Z); the adjustment set Z *is* the conditioning variables.
- **LLM causal-reasoning benchmarks (Corr2Cause, CLadder 2023–25)** — tasks ask models to pick a valid backdoor set / detect colliders; models pattern-match the rule rather than apply d-separation.
- **Off-policy eval & RecSys debiasing** — adjusting logged-data confounders (context that drove both action and reward) is a backdoor set.

## Common mistake

"Adjust for everything to be safe." Conditioning on a **collider** (or its descendant) *opens* a previously-blocked path and **creates** bias (collider/M-bias); conditioning on a **mediator** (descendant of X on the causal path) blocks part of the effect you want. More controls is not safer — block back-door paths only, never causal paths, colliders, or post-treatment variables.

## See also
- [[do-calculus]] — backdoor is the special case; full ID handles effects no single adjustment set identifies
- [[frontdoor-criterion-mediation-analysis]] — the fallback when an unblockable back-door path exists but a clean mediator does
- [[propensity-scores-ipw]] — reweighting on e(Z)=P(X=1|Z) achieves the same adjustment via the propensity instead of the outcome model
