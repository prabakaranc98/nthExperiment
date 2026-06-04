# Evals Are Experiments

*Why bolding the bigger number without a confidence interval is scientifically invalid.*

---

## The problem

A leaderboard says: Model A scores 73.2%, Model B scores 72.8%. Is A better?

**You can't tell.** That 0.4-point gap may be noise. Without the variance, the claim is unfalsifiable.

This remains common in 2025-2026 reporting: point estimates with no confidence intervals, no statistical tests, no power analysis. Many headline "improvements" sit inside the margin of noise.

The fix is a reframe: **an eval is a statistical experiment, not a scoreboard.** Treat it like one.

---

## Evals as statistical experiments

The core idea (Miller, Anthropic 2024 — *Adding Error Bars to Evals*): **treat benchmark questions as a random sample from a larger population of questions.**

Your 1,000 benchmark questions estimate the model's performance on the *population* of all possible questions of that type. That estimate carries uncertainty — quantify it.

### Confidence interval for accuracy

Each question is a coin flip: pass (1) or fail (0). With `n` questions and `k` correct:

| Quantity | Formula |
|---|---|
| Point estimate | p̂ = k / n |
| Standard error | SE = √(p̂(1−p̂) / n) |
| 95% CI (approx) | [p̂ − 1.96·SE, p̂ + 1.96·SE] |

**Example.** 73.2% on 1,000 questions → SE ≈ √(0.732 × 0.268 / 1000) ≈ 1.4% → 95% CI ≈ [70.4%, 76.0%]. A model at 72.8% sits squarely inside that interval.

---

## The paired test — comparing two models properly

The naive comparison computes a CI for each model separately and checks overlap. **A paired test is stronger.** If both models answered the *same questions*, you can look at *which questions* one got right and the other got wrong, cancelling out per-question difficulty.

**McNemar's test** (for paired binary outcomes) ignores the agreements and tests only the disagreements, giving much higher power — often 5-10× fewer questions to detect the same gap.

**Design rule:** run both models on identical question sets so you can pair.

---

## When not to use the CLT

The normal approximation above relies on the Central Limit Theorem and assumes `n` is large enough. Use exact or Bayesian methods instead when:

- **n < 100** — CLT is unreliable; use exact binomial (Clopper-Pearson) CIs.
- **Clustered data** — multiple questions per topic/document violate independence; ignoring clustering underestimates variance.
- **Short, specialized benchmarks** — e.g. GPQA Diamond (~200 questions); prefer Bayesian credible intervals or exact methods.

Bowyer et al. (2025): don't blindly apply the CLT to LLM evals on small datasets.

---

## Power analysis — planning evals before you run them

Power analysis answers: *how many questions do I need to reliably detect a gap of Δ?*

**Ingredients:**

- **Effect size** Δ — the gap you want to detect.
- **Significance level** α — typically 0.05.
- **Power** 1−β — typically 0.8 (80% chance of detecting a real effect).

For proportions, a rough sizing formula:

```
n ≈ 2 · (z_{α/2} + z_β)² · p̄(1−p̄) / Δ²
```

where p̄ is the average accuracy and the `z` terms are normal quantiles.

**Implication:** detecting a 1-point improvement at 80% power, α=0.05 needs ~10,000+ questions. Most benchmarks are badly underpowered for the small gaps people report.

---

## The Chatbot Arena approach

Chatbot Arena / LMArena (Chiang et al., 2023) ranks models from pairwise human preferences using a **Bradley-Terry model**: each model has a latent strength, and

```
P(A beats B) = σ(s_A − s_B) = exp(s_A) / (exp(s_A) + exp(s_B))
```

Strengths are fit by maximum likelihood; confidence intervals come from bootstrapping.

**The leaderboard's own lesson:** top models routinely have overlapping CIs. Rank 1 and rank 5 may not be distinguishable. The rank number is not the truth — read the intervals.

---

## Summary: what to do

When reporting eval results:

1. **Always report a CI**, not just a point estimate.
2. **Use paired tests** (e.g. McNemar's) when comparing models on the same questions.
3. **Use exact or Bayesian methods** when n < 200 or data is clustered.
4. **Run a power analysis first** when you want to detect a specific gap.
5. **Separate statistical from practical significance** — a real but tiny gain may not justify the deployment cost.

---

*Related: see the [concept-library index](../bricks/README.md) for adjacent topics on measurement and uncertainty.*
