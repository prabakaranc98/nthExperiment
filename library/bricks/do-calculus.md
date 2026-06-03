# DAGs & Do-Calculus

**One-liner:** Pearl's framework — a causal DAG plus three rewrite rules that turn an interventional query P(Y | do(X)) into estimable observational quantities; the backdoor criterion is the workhorse special case.

## The setup

A causal DAG G encodes which variables directly cause which (edges = direct causal effects). `do(X=x)` is the *intervention* operator: it surgically deletes all incoming edges to X and sets X=x, defining a new graph G_X̄. The interventional distribution P(Y | do(X=x)) is generally **not** the observational P(Y | X=x) — confounding makes them differ.

## The three rules

For disjoint node sets X, Y, Z, W, do-calculus rewrites do-expressions using d-separation (⊥⊥) in surgically-modified graphs:

1. **Insert/delete observation:** P(y | do(x), z, w) = P(y | do(x), w) if (Y ⊥⊥ Z | X, W) in G_X̄
2. **Action/observation exchange:** P(y | do(x), do(z), w) = P(y | do(x), z, w) if (Y ⊥⊥ Z | X, W) in G_X̄Z̲
3. **Insert/delete action:** P(y | do(x), do(z), w) = P(y | do(x), w) if (Y ⊥⊥ Z | X, W) in G_X̄Z(W)‾

Where X̄ = delete edges *into* X, Z̲ = delete edges *out of* Z. **Completeness (Shpitser–Pearl, Huang–Valtorta):** these three rules + probability axioms identify every identifiable effect; if no derivation exists, the effect is non-identifiable from observational data + G.

## The backdoor criterion (the case you actually use)

Z satisfies the backdoor criterion for (X→Y) if (a) no node in Z is a descendant of X, and (b) Z blocks every path from X to Y with an arrow *into* X. Then:

P(Y | do(x)) = Σ_z P(Y | x, z) P(z)   — the **adjustment formula**

When backdoor fails, the **frontdoor criterion** or full do-calculus (the ID algorithm) may still identify the effect via a mediator.

## Where it appears

- **ID algorithm / `causaleffect`, DoWhy, Ananke** — automated do-calculus: input DAG + query, output an estimand or "non-identifiable"
- **Double/debiased ML (Chernozhukov)** — estimates the backdoor adjustment with ML nuisance models + Neyman-orthogonal scores
- **LLM causal reasoning (2024-2026 benchmarks: Corr2Cause, CLadder)** — models asked to apply backdoor/d-separation; they pattern-match more than reason
- **RL & off-policy eval** — do(action) is exactly the interventional/policy distribution; confounded logged data needs adjustment

## Common mistake

Conditioning on a **collider** (or a descendant of one) to "control for everything." Adjusting for a common *effect* of X and Y opens a spurious path and *creates* bias — more controls is not safer. The backdoor set must block back-door paths, not all paths, and must contain no colliders or descendants of X.

## See also
- [[potential-outcomes]] — the Neyman–Rubin counterfactual view; do(x) ≈ setting the treatment, backdoor ≈ ignorability/no-unmeasured-confounders
- [[causal-representation-learning]] — recovering the DAG's variables themselves from raw data when nodes aren't given
- [[ppi]] — valid downstream inference once an estimand is identified
