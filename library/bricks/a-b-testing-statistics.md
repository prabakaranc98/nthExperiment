# A/B Testing Statistics (Power, MDE, Multiple Comparisons)

**One-liner:** Sizing experiments so a real effect of size MDE is detectable with power 1−β at level α, shrinking variance via pairing/CUPED, and controlling family-wise error or FDR when comparing many models/metrics at once — the antidote to "model B scored 0.3 higher, so it's better."

## The formulas

**Sample size (two-sample, per arm)** for two-sided level α, power 1−β:

n ≈ 2σ² (z_{1−α/2} + z_{1−β})² / Δ²

where Δ = MDE (minimum detectable effect) and σ² the metric variance. For α=0.05, β=0.2 the constant (z+z)² ≈ 7.85. Effect size in σ units: d = Δ/σ.

**MDE (invert for fixed n):** Δ = (z_{1−α/2} + z_{1−β}) · σ · √(2/n).

**Variance reduction (CUPED):** Y' = Y − θ(X − E[X]), θ = Cov(Y,X)/Var(X) with X a pre-experiment covariate. Var(Y') = Var(Y)(1−ρ²) ⇒ n shrinks by (1−ρ²). Paired/within-prompt designs do the same: test the diff per item, not group means.

**Multiple comparisons** (m tests):
- Bonferroni: reject if p_i ≤ α/m (controls FWER, conservative).
- Benjamini–Hochberg: sort p_(1)≤…≤p_(m), reject largest i with p_(i) ≤ (i/m)α (controls FDR, the usual choice for many metrics/checkpoints).

## Where it appears

- **LLM eval harnesses (lm-eval-harness, HELM, Open LLM Leaderboard)** — per-task accuracy diffs need paired bootstrap CIs + BH across the suite, not a raw leaderboard delta.
- **Chatbot Arena / Elo** — head-to-head with confidence intervals; "rank N is indistinguishable from rank N+3" is a power/multiplicity statement.
- **Industrial A/B platforms (CUPED at Microsoft/Netflix)** — covariate adjustment to hit MDE on huge-variance engagement metrics.
- **RLHF / preference eval** — win-rate is a paired Bernoulli; n for a 2-point win-rate move is large.

## Common mistake

"Peeking": checking significance repeatedly and stopping at first p<0.05 inflates the false-positive rate far above α. Fixed-horizon tests assume one look at the planned n. To monitor continuously, use sequential/always-valid methods, not naive repeated t-tests. Equally common: reporting a single number with no CI and no power analysis, then declaring a winner on noise.

## See also
- [[bootstrap-confidence-intervals-for-eval-metrics]] — how to get the CI/variance the power calc needs
- [[sequential-testing-always-valid-p-values]] — the correct fix for peeking / continuous monitoring
- [[elo-online-rating-for-model-ranking]] — ranking many models where multiplicity bites hardest
