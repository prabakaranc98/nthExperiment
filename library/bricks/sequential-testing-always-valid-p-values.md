# Sequential Testing & Always-Valid p-Values

**One-liner:** Anytime-valid inference (e-values, mSPRT, confidence sequences) where the Type-I error guarantee holds at every sample size simultaneously — so you can peek continuously and stop the moment you see significance without inflating false-positive rates, the modern fix for "peeking" in rolling A/B tests and LLM evals.

## The key insight

A fixed-n test's α guarantee evaporates if you check it repeatedly (optional stopping inflates Type-I error toward 1). The fix: build a test statistic that is a **nonnegative supermartingale** under H₀, then apply Ville's inequality.

**E-value / e-process:** E_t ≥ 0 with E_{H₀}[E_t] ≤ 1 (a martingale wealth process betting against H₀). Ville's inequality gives the anytime guarantee:

  P_{H₀}( ∃t : E_t ≥ 1/α ) ≤ α

So the **always-valid p-value** p_t = 1 / max_{s≤t} E_s ≥ inf over a peeking rule, and rejecting the first time p_t ≤ α controls Type-I error over the *entire* stopping horizon. E-values multiply across independent experiments and combine by averaging — closed under optional continuation.

**Mixture SPRT (Robbins):** likelihood ratio Λ_t(θ) mixed over a prior, ∫ Λ_t(θ) dπ(θ), is a martingale → no need to pre-specify the alternative.

**Confidence sequence:** an interval CI_t with P(∀t: θ ∈ CI_t) ≥ 1−α (vs. a fixed CI valid only at one n); width shrinks ~√(log log t / t) (law of iterated logarithm rate, the price of anytime validity).

## Where it appears

- **Industrial A/B testing** — Netflix, Optimizely, Evidently, Eppo ship confidence sequences / mSPRT so PMs can stop early without p-hacking
- **Rolling LLM evals** — stream eval examples, stop once model A beats B at anytime-valid α; avoids fixing eval-set size in advance, handles continuous monitoring of deployed models
- **Off-policy / RLHF gating** — e-process tests for "is the new policy better" under continuous monitoring
- **Game-theoretic statistics** — Vovk & Shafer "testing by betting"; Ramdas et al. (2023) game-theoretic survey of e-values

## Common mistake

Reporting the e-value or anytime-valid p_t but then doing a *fixed-n* power calculation, or interpreting the running p_t like a classical one. Also: thinking confidence sequences are free — they are strictly wider than a fixed-n CI at any single n (the √(log log t) penalty), so for a known, pre-committed sample size a fixed test is more powerful.

## See also
- [[a-b-testing-statistics]] — the classical fixed-horizon framework this replaces for streaming/peeking
- [[bootstrap-confidence-intervals-for-eval-metrics]] — fixed-sample eval uncertainty vs. anytime-valid sequences
- [[ppi]] — combine with prediction-powered inference for valid sequential testing on ML-labeled eval data
