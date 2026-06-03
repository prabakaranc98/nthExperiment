# Epistemic vs Aleatoric Uncertainty

**One-liner:** Decompose predictive uncertainty into epistemic (reducible — model/knowledge uncertainty, shrinks with data) and aleatoric (irreducible — inherent data noise, stays even with infinite data); the split drives selective prediction, hallucination detection, and active learning.

## The decomposition

For a Bayesian predictive p(y|x) = ∫ p(y|x,θ) p(θ|D) dθ, total uncertainty splits via the law of total variance / entropy:

H[y|x,D] = E_{θ~p(θ|D)} H[y|x,θ]  +  I[y; θ | x, D]
&nbsp;&nbsp;(total)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = aleatoric &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+&nbsp; epistemic

- **Aleatoric** = expected entropy of the per-parameter predictive = irreducible noise (averaged over models, each model is still uncertain).
- **Epistemic** = mutual information between y and θ = how much models *disagree*. Vanishes as p(θ|D) concentrates (more data, less ambiguity).

For regression with predicted mean μ and variance σ²(x): aleatoric ≈ E[σ²], epistemic ≈ Var[μ] across the posterior/ensemble.

## Where it appears

- **Deep ensembles / MC-dropout** — approximate the posterior with M samples; epistemic = disagreement (variance of means or BALD = I[y;θ]), the basis for OOD detection.
- **Semantic entropy (Kuhn et al. 2023; Farquhar et al., Nature 2024)** — clusters LLM samples by meaning; high entropy across meanings flags *epistemic* uncertainty → confabulation/hallucination.
- **Selective prediction / abstention** — reject when epistemic is high (model doesn't know); high aleatoric alone is no reason to abstain (irreducible).
- **Active learning (BALD)** — acquire points of maximal *epistemic* uncertainty; querying high-aleatoric points wastes labels.
- **Heteroscedastic regression** — network outputs σ²(x) directly as a learned aleatoric head (NLL loss).

## Common mistake

Treating high softmax entropy / low max-prob as "epistemic." A single forward pass gives you *total* (mostly aleatoric) uncertainty — a confidently-wrong model and a genuinely-ambiguous input look identical. Isolating epistemic requires *multiple* hypotheses (ensemble, posterior samples, MC-dropout, or sampled generations) and measuring their *disagreement*.

## See also
- [[deep-ensembles-mc-dropout]] — the standard practical estimator of epistemic uncertainty
- [[semantic-entropy-for-hallucination-detection]] — epistemic uncertainty over meanings to catch LLM confabulation
- [[calibration]] — well-calibrated total probabilities are a prerequisite for trusting either component
