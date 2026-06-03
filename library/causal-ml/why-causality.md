# Why Causality Matters

*The gap between correlation and causation — and why it breaks ML.*

---

## The core problem

Statistical ML finds patterns. Causality asks: **why** do those patterns exist?

**Famous example: ice cream and drowning.** Ice cream sales and drowning deaths are highly correlated. Should you ban ice cream to prevent drowning?

No. Both are caused by a **confounder**: hot weather. Hot weather → more ice cream sales. Hot weather → more swimming → more drowning.

A model trained on this data would predict: selling less ice cream reduces drowning. This prediction fails when you *intervene* (actually cut ice cream sales in winter — drowning doesn't change).

---

## The ladder of causation (Pearl)

Three levels of causal reasoning, each strictly more powerful than the last:

| Level | Question | Example |
|-------|----------|---------|
| **Association (L1)** | What is? What correlates? | *What is P(Y \| X = x)?* |
| **Intervention (L2)** | What if I do X? | *What is P(Y \| do(X = x))?* |
| **Counterfactual (L3)** | What if I had done X instead? | *What would Y have been if X had been different?* |

Statistical ML lives at L1. Causal inference reaches L2 and L3.

**The crucial difference:** *P(Y | X = x)* is observed correlation. *P(Y | do(X = x))* is the effect of *intervening* and forcing X = x, regardless of what caused X before. These can be completely different numbers.

---

## Why ML breaks under distribution shift

A model trained on historical data learns P(Y | X) — the correlation in the training distribution. When the distribution shifts (different population, new policy, new environment), this correlation may not hold.

**Causal features are invariant.** The mechanism "fire causes smoke" doesn't change when you move to a different country. The spurious correlation "smoke detector → smoke" in the training data *does* change when you move to a building with better fire prevention.

This is the core insight behind **Invariant Risk Minimization (IRM)** and **Causal Representation Learning**: learn features whose relationship to the target is *mechanistic* (invariant across environments), not spurious (environment-specific).

---

## Why this matters for your work

1. **Decision engineering:** making a decision requires L2 reasoning (what happens if I *do* X?), not L1 (what does X correlate with?). A model that predicts outcomes based on correlations gives you wrong answers when you use it to make decisions.

2. **CRL × continual learning:** if a model learns spurious features, it will fail under distribution shift. If it learns causal features (the generating mechanisms), it should generalize across environments.

3. **Foundation models and causal structure:** do large pretrained models implicitly learn causal structure? Can you probe them for it? This is an active frontier (TabPFN's causal prior is one data point).

---

## The key distinctions to internalize

| Statistical | Causal |
|------------|--------|
| P(Y \| X) | P(Y \| do(X)) |
| Correlation | Mechanism |
| Breaks under shift | Invariant across environments |
| L1 reasoning | L2/L3 reasoning |
| "What is?" | "What if I do X?" |

**The fundamental asymmetry:** you can always compute P(Y|X) from data. You *cannot* always compute P(Y|do(X)) from data — it requires either randomized experiments or additional assumptions about the causal structure.
