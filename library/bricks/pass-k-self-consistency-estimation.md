# pass@k & Self-Consistency Estimation

**One-liner:** pass@k estimates the probability at least one of k samples is correct (unbiased estimator from n≥k draws), while self-consistency reports the accuracy of the majority vote over sampled chains — the two standard ways to turn many stochastic samples into one code/math/reasoning score.

## The formula / definition

**Unbiased pass@k (Codex / HumanEval, Chen et al. 2021).** Draw n ≥ k samples per problem, count c correct. Estimate per problem:

  pass@k = E[ 1 − C(n−c, k) / C(n, k) ]   (averaged over problems)

This is the chance that a random size-k subset of the n samples contains ≥1 correct one. Compute it in the log/product form to avoid overflow:

  1 − ∏_{i=n−c+1}^{n} (1 − k/i)

The naive estimator (1 − (1 − p̂)^k with p̂ = c/n) is **biased**; use the combinatorial form. Requires n > k for variance reduction (typical: n=200, report pass@1, pass@10, pass@100).

**Self-consistency (Wang et al. 2022).** Sample m chains-of-thought at temperature T>0, marginalize over reasoning by majority-voting the *final answers*:

  ŷ = argmax_a Σ_{i=1}^{m} 1[answer(chain_i) = a]

Accuracy = 1[ŷ = y*]. Weighted variants vote by verifier/RM score or log-prob.

## Where it appears

- **HumanEval / MBPP / SWE-bench** — pass@k is the canonical code-gen metric; pass@1 with greedy vs. pass@k with sampling.
- **GSM8K / MATH** — self-consistency (maj@m) is the default reasoning-accuracy lift; o1/R1-style models report cons@64.
- **RLVR & best-of-n** — pass@k upper-bounds what an outcome-verifier or RL policy can extract; the pass@1↔pass@k gap motivates verifier search.
- **Test-time scaling curves** — accuracy plotted vs. m or k to show sample-efficiency of test-time compute.

## Common mistake

Conflating the two: **pass@k is an oracle/verifier metric** (any one sample correct — requires a checker), while **self-consistency/maj@k is selection-free** (commits to the vote, no ground-truth access at inference). pass@k ≥ maj@k always, and the gap is exactly the headroom a good verifier buys you. Also: using the biased 1−(1−p̂)^k estimator instead of the combinatorial one.

## See also
- [[self-consistency]] — the majority-vote half of this card, in full
- [[rejection-sampling-best-of-n]] — pass@k is the verifier-oracle ceiling that best-of-n tries to reach
- [[compute-optimal-test-time-allocation]] — how many samples k to spend per problem
