# Why Causality Matters

*The gap between correlation and causation — and why it breaks ML.*

---

## The core problem

Statistical ML finds patterns. Causality asks a different question: **why** do those patterns exist?

**The classic example: ice cream and drowning.** Ice cream sales and drowning deaths are highly correlated. Should you ban ice cream to prevent drowning?

No. Both are driven by a **confounder** — hot weather:

- Hot weather → more ice cream sales
- Hot weather → more swimming → more drowning

A model trained on this data would predict that cutting ice cream sales reduces drowning. That prediction fails the moment you *intervene*: stop selling ice cream in winter and drowning doesn't budge. The correlation was real; the causal claim was not.

---

## The ladder of causation (Pearl)

Three levels of reasoning, each strictly more powerful than the last:

| Level | Question | Formal form |
|-------|----------|-------------|
| **Association (L1)** | What is? What correlates? | *P(Y \| X = x)* |
| **Intervention (L2)** | What if I *do* X? | *P(Y \| do(X = x))* |
| **Counterfactual (L3)** | What if I *had done* X instead? | *What would Y have been had X differed?* |

Statistical ML lives at L1. Causal inference reaches L2 and L3.

**The crucial difference:** *P(Y \| X = x)* is observed correlation — you select the cases where X happened to equal x. *P(Y \| do(X = x))* is the effect of forcing X = x by intervention, regardless of what set X before. These can be completely different numbers.

---

## Why ML breaks under distribution shift

A model trained on historical data learns *P(Y \| X)* — the correlation in the *training* distribution. When that distribution shifts (new population, new policy, new environment), the correlation can vanish.

**Causal features are invariant; spurious ones are not.**

- "Fire causes smoke" holds in any country, any building.
- "Smoke detector → smoke" holds only in the training data. Move to a building with better fire prevention and that correlation breaks.

This is the core insight behind **Invariant Risk Minimization (IRM)** and **causal representation learning**: prefer features whose relationship to the target is *mechanistic* (invariant across environments) over those that are *spurious* (environment-specific). See the [concept-library index](../bricks/README.md) for related entries.

---

## Why this matters for your work

- **Decision engineering.** Making a decision is an L2 question — *what happens if I do X?* — not an L1 one. A model that ranks options by correlation gives wrong answers the instant you act on it.
- **CRL × continual learning.** A model that latches onto spurious features fails under shift. One that recovers the generating mechanisms should transfer across environments — exactly what continual learning needs.
- **Foundation models and causal structure.** Do large pretrained models implicitly encode causal structure, and can you probe them for it? Still an active frontier in 2025-2026 — work on causal scrubbing, mechanistic interpretability, and structured tabular priors (e.g. TabPFN's prior-data design) are data points, not settled answers.

---

## Key distinctions to internalize

| Statistical | Causal |
|-------------|--------|
| *P(Y \| X)* | *P(Y \| do(X))* |
| Correlation | Mechanism |
| Breaks under shift | Invariant across environments |
| L1 reasoning | L2 / L3 reasoning |
| "What is?" | "What if I do X?" |

**The fundamental asymmetry:** you can *always* compute *P(Y \| X)* from data. You *cannot* always compute *P(Y \| do(X))* from data — that requires either randomized experiments or explicit assumptions about the causal structure.
