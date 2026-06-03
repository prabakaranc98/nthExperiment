# Evals Are Experiments

*Why bolding the bigger number without a confidence interval is scientifically invalid.*

---

## The problem

A benchmark says: Model A gets 73.2%, Model B gets 72.8%. Is A better?

**You don't know.** That 0.4% difference might be noise. Without knowing the variance, you can't make a valid claim.

This is the norm in AI: papers report point estimates with no confidence intervals, no statistical tests, no power analysis. Most published "improvements" are within the margin of noise.

---

## Evals as statistical experiments

The key reframe (Miller, Anthropic 2024): **treat benchmark questions as a random sample from a larger population of questions**.

Your sample of 1000 benchmark questions estimates the model's performance on the *population* of all possible questions of this type. The estimate has uncertainty, and you should quantify it.

**How to get a confidence interval for eval accuracy:**

The questions are like coin flips: pass (1) or fail (0). With n questions and k correct:

- Point estimate: p̂ = k/n
- Standard error: SE = √(p̂(1-p̂)/n)
- 95% CI (approx): [p̂ - 1.96·SE, p̂ + 1.96·SE]

**Example:** 73.2% on 1000 questions → SE = √(0.732 × 0.268 / 1000) ≈ 1.4% → 95% CI: [70.4%, 76.0%]. A model with 72.8% is well within this interval.

---

## The paired test — comparing two models properly

The naive comparison computes CIs for each model separately. **The paired test is better**: if both models answered the *same questions*, you can measure *which questions* one got right and the other got wrong.

**McNemar's test** (for paired binary outcomes) has much higher power. Instead of needing the full variance, you only look at the disagreements. This can require 5–10× fewer questions to detect the same gap.

**Lesson:** design evals to be paired — same questions for both models.

---

## When not to use the CLT

The normal approximation (SE formula above) relies on the Central Limit Theorem and assumes n is large enough. For:
- n < 100: CLT is unreliable; use exact binomial CIs
- Clustered data (multiple questions from the same topic): ignore clustering and you underestimate variance
- Short, specialized benchmarks (e.g., GPQA with ~200 questions): Bayesian credible intervals or exact methods

Bowyer et al. (2025): "Don't blindly use the CLT in LLM evals with small datasets."

---

## Power analysis — planning evals properly

Power analysis answers: how many questions do I need to reliably detect a difference of Δ?

**Ingredients:**
- Effect size: Δ (the gap you want to detect)
- Significance level: α (typically 0.05)
- Power: 1-β (typically 0.8 — 80% chance of detecting the effect if real)

For proportions, a rough formula: n ≈ 2 × (z_{α/2} + z_β)² × p̄(1-p̄) / Δ²

where p̄ is the average accuracy and z values are normal quantiles.

**Lesson:** if you want to detect a 1% improvement with 80% power at α=0.05, you need ~10,000+ questions. Most benchmarks are massively underpowered for small improvements.

---

## The Chatbot Arena approach

Chatbot Arena (Chiang et al., 2023) uses pairwise human preferences and fits a **Bradley-Terry model**: each model has a latent strength, and the probability of model A beating model B is:

```
P(A beats B) = σ(strength_A - strength_B) = exp(s_A) / (exp(s_A) + exp(s_B))
```

Strengths are estimated by maximum likelihood. Confidence intervals come from bootstrapping.

**The key takeaway from the leaderboard:** top models often have overlapping confidence intervals. Rank 1 and rank 5 might not be distinguishably different. The rank number is not the truth.

---

## Summary: what to do

When reporting eval results:
1. **Always report a CI**, not just a point estimate
2. **Use paired tests** when comparing two models on the same questions
3. **Use exact/Bayesian methods** when n < 200
4. **Run a power analysis first** if you want to detect a specific gap
5. **Distinguish statistical significance from practical significance** — a real but tiny improvement may not be worth the deployment cost
