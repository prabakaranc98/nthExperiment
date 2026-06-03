# Measure Theory & Probability Limits

*Concentration inequalities, CLT failure modes, exchangeability — the formal foundations of evaluation, conformal prediction, and generalization bounds.*

---

## Why this matters at the frontier

- **Confidence intervals for evals** require knowing when the Central Limit Theorem holds — and when it doesn't. Using CLT-based intervals on 50-question benchmarks gives wildly wrong error bars.
- **Conformal prediction** requires *exchangeability* — a weaker assumption than i.i.d. that you need to understand to know when the guarantee holds.
- **Generalization bounds** (PAC-Bayes, Rademacher) are stated in terms of probability over the training set. Concentration inequalities are the proof tools.
- **E-values and anytime-valid inference** are built on measure-theoretic probability — you need to know what it means for a process to be a martingale.

---

## Measure theory in one page (what you actually need)

A **measure** μ on a set Ω is a function that assigns a non-negative number to subsets of Ω, satisfying countable additivity. A **probability measure** has μ(Ω) = 1.

**Why measure theory for ML?** Standard probability assumes you can enumerate outcomes (discrete) or integrate nicely (continuous). Measure theory handles both uniformly and rigorously handles limiting processes (what happens as n → ∞?).

**Three things that matter:**
1. **Almost surely (a.s.) vs. in probability vs. in L²:** three types of convergence with different strengths. Most ML theorems use "in probability" or "almost surely."
2. **Expectation = integral w.r.t. probability measure.** E[f(X)] = ∫ f(x) dP(x). Change of measure = Radon-Nikodym derivative.
3. **Conditional expectation** is not just "average given event" — it's a function that satisfies a tower property. This is the foundation for martingales.

---

## Concentration inequalities — the proof tools for generalization

These bound the probability that a random variable deviates from its mean.

**Markov's inequality:** P(X ≥ t) ≤ E[X]/t for X ≥ 0.

**Chebyshev's inequality:** P(|X - μ| ≥ t) ≤ Var[X]/t².

**Hoeffding's inequality:** for independent bounded random variables Xᵢ ∈ [aᵢ, bᵢ]:
```
P(|X̄ - E[X̄]| ≥ t) ≤ 2 exp(-2n²t² / Σᵢ(bᵢ-aᵢ)²)
```

This is the workhorse. For {0,1}-valued variables (like evaluation accuracy), each Xᵢ ∈ [0,1], so:
```
P(|X̄ - p| ≥ t) ≤ 2 exp(-2nt²)
```

Setting this ≤ δ and solving for t: **t = √(log(2/δ)/(2n))**

This gives a valid confidence bound for evaluation accuracy that's free of CLT assumptions. For n=100, δ=0.05: t = √(log(40)/200) ≈ 0.135. So a 95% CI has width ~27% — huge.

**Bernstein's inequality:** tighter when variance is small (close to 0 or 1 accuracy):
```
P(X̄ - p ≥ t) ≤ exp(-nt² / (2(p(1-p) + t/3)))
```

**Sub-Gaussian variables:** a class that generalizes Gaussian — Hoeffding's inequality applies to any sub-Gaussian variable. Standard Gaussians, bounded variables, and their sums are all sub-Gaussian.

---

## The CLT and its failure modes

**Central Limit Theorem:** if X₁, ..., Xₙ are i.i.d. with mean μ and variance σ², then:
```
√n(X̄ - μ)/σ → N(0,1) as n → ∞
```

**The normal approximation** gives CI: μ ∈ [X̄ ± z_{α/2} · σ/√n] ≈ valid for large n.

**When it fails:**
1. **n is small** (< 100): convergence to normal is slow. The approximation error can be substantial, especially for binary outcomes far from 0.5.
2. **Heavy-tailed distributions:** if variance is infinite (or very large), CLT convergence is extremely slow. The Berry-Esseen theorem gives the rate: approximation error ≤ C · E[|X|³] / (σ³ √n).
3. **Dependence / clustering:** if questions in a benchmark cluster by topic, the effective sample size is less than n. Using n inflates your confidence.
4. **Non-i.i.d. test data:** if the test set has systematic biases (e.g., all examples from one domain), CLT doesn't apply at all.

**Bowyer et al. (2025):** show that on specialized AI benchmarks with n < 200, CLT-based intervals can understate uncertainty by 2-3× relative to exact or Bayesian intervals. Use binomial exact CIs or Bayesian beta-binomial posteriors for small n.

---

## Exchangeability — the foundation of conformal prediction

**i.i.d.** is a strong assumption: samples are both *independent* and *identically distributed*.

**Exchangeability** is weaker: a sequence X₁, ..., Xₙ is exchangeable if its distribution is unchanged by any permutation of indices.

- i.i.d. → exchangeable (trivially)
- Exchangeable does *not* imply i.i.d. (e.g., a random sample without replacement from a fixed population is exchangeable but not independent)

**Why conformal prediction uses exchangeability:** the split conformal guarantee is:

P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1-α

*as long as* (X₁,Y₁), ..., (Xₙ,Yₙ), (X_{n+1}, Y_{n+1}) are exchangeable (not necessarily i.i.d.).

This is valid for random test points, for leave-one-out settings, and for many natural experimental designs where i.i.d. would be too strong.

**When conformal validity breaks:** if the test point has a *different distribution* from calibration data (covariate shift), exchangeability fails. Tibshirani, Barber, Candès, Ramdas (2019) fix this using importance weighting.

---

## Martingales and anytime-valid inference

A **martingale** is a stochastic process Mₜ (indexed by time t) satisfying:
```
E[Mₜ₊₁ | M₁, ..., Mₜ] = Mₜ
```

A martingale has no trend — its expected future value equals its current value. It's a fair game.

**Optional Stopping Theorem:** if τ is a stopping time (a time you decide to stop based on what you've seen so far), then E[M_τ] = E[M₁] under mild conditions. This is why classical hypothesis tests break when you peek and stop early — you've turned a fair game unfair.

**E-values (Vovk & Wang):** an e-value is a random variable E with E[E] ≤ 1 under the null. If E is large, the null is unlikely. Unlike p-values, e-values can be *multiplied* across independent tests to form a combined test. The product M_τ = Π_t E_t is a martingale under the null.

**Anytime-valid confidence sequences:** a CI function C(X₁,...,Xₜ) that satisfies:
```
P(∀t ≥ 1: μ ∈ C(X₁,...,Xₜ)) ≥ 1-α
```

— it's valid at *every* time point simultaneously, not just at one pre-specified n. This is what you need for continuous monitoring and adaptive stopping.

**How it's built:** via the "method of mixtures" (Howard, Ramdas et al.) — construct a mixture of martingales that concentrates around the mean while being valid at all sample sizes.

**Chatbot Arena uses E-values** for its sequential update of model rankings — as new preferences arrive, they update rankings in a way that maintains valid CIs at every step.

---

## Rademacher complexity — the right complexity measure

For a function class F (e.g., neural networks with bounded weights):

**Empirical Rademacher complexity:**
```
R̂_n(F) = E_σ[sup_{f∈F} (1/n) Σᵢ σᵢ f(xᵢ)]
```

where σᵢ ∈ {-1, +1} are independent Rademacher random variables (uniform ±1 coin flips).

**Intuition:** if F can fit random noise well (large R̂_n(F)), it's complex and prone to overfitting. If it can't correlate with random labels, it's well-controlled.

**Generalization bound:**
```
E[L(f)] ≤ L̂(f) + 2R_n(F) + O(√(log(1/δ)/n))
```

**Why it's better than VC dimension:** it's data-dependent. If your actual training data happens to be easy (concentrated distribution), the empirical Rademacher complexity is small even for a high-capacity function class.

**Frontier relevance:** understanding why empirical risk minimization works for LLMs requires Rademacher complexity or PAC-Bayes bounds — VC dimension is useless in the overparameterized regime.
