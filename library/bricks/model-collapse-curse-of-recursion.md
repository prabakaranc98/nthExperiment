# Model Collapse / Curse of Recursion

**One-liner:** Training generation n+1 on the outputs of generation n drives a degenerative feedback loop — tails are forgotten first, variance shrinks, then the mean drifts — until the model converges to a low-entropy point mass; mixing/accumulating real data with synthetic bounds the damage.

## The key insight

Each generation re-estimates a distribution from a finite sample of the previous model's outputs. Three compounding errors accumulate (Shumailov et al., *Nature* 2024): (1) **statistical error** — finite-sample noise loses low-probability events (tails) first; (2) **functional expressivity** error — finite model capacity; (3) **functional approximation** error — optimizer/SGD bias.

Clean Gaussian case: sample N points from N(mu, sigma^2), refit, repeat. Variance follows a multiplicative random shrink each round, so it decays toward 0 and the estimated mean executes a random walk that almost surely diverges as t -> infinity:

```
sigma_t^2  ->  0      (diversity collapses)
Var(mu_t)  =  sigma_0^2 * sum_{i<t} 1/N_i   (mean drifts, unbounded)
```

The fix depends on the data regime (Gerstgrasser et al. 2024; Dohmatob et al. 2024):
- **Replace** real data with synthetic each round -> error grows linearly in n; collapse.
- **Accumulate** (real + all prior synthetic) -> test error is *bounded* by a finite constant independent of n. The original real data acts as an anchor.

## Where it appears

- Shumailov et al. (*Nature* 2024) — coined "model collapse"; showed LLMs/VAEs/GMMs degenerate under recursive self-training.
- Gerstgrasser et al. 2024 ("accumulate, don't replace") — proves accumulating data caps the error; the practical guardrail for synthetic-data pipelines.
- Synthetic-data pretraining (Phi, Llama-3 distillation, web rephrasing) — why labs anchor on real corpora and filter, rather than train purely on generations.
- Dead-internet / web-feedback concern — future crawls contain LLM text, so uncontrolled "replace" dynamics happen implicitly at corpus scale.

## Common mistake

Believing synthetic data is inherently poison. It is not — collapse is driven by *replacing* real data and by *unfiltered* recursion. Accumulating real + synthetic, or keeping a fixed real anchor with quality filtering, bounds the error and synthetic data remains a net win. Also: collapse hits the tails (rare modes, minorities) long before average perplexity looks bad, so mean-metric monitoring misses it.

## See also
- [[synthetic-data-web-rephrasing]] — the production technique whose safety this bounds
- [[data-constrained-scaling-repetition-laws]] — the data-scarcity pressure that makes synthetic generation tempting
- [[memorization-vs-generalization]] — tail forgetting vs collapse to memorized modes
