# PAC-Bayes

**One-liner:** Generalization bounds on the *expected* risk of a stochastic (Gibbs) classifier, controlled by KL(Q‖P) between a learned posterior Q and a data-independent prior P over hypotheses — the only framework to date that yields non-vacuous bounds for overparameterized nets.

## The classic bound (McAllester, 1999)

For any prior P (chosen before seeing data), with probability ≥ 1−δ over the sample S (size m), simultaneously for all posteriors Q:

  𝔼_{h∼Q}[R(h)] ≤ 𝔼_{h∼Q}[R̂_S(h)] + √( (KL(Q‖P) + ln(2√m/δ)) / (2m) )

R = true risk, R̂_S = empirical risk. The complexity term is KL(Q‖P)/m, **not** a parameter count. Tighter forms: Catoni (2007), Maurer's bound, and the binary-KL (Seeger/Langford) inversion kl(R̂‖R) ≤ (KL+ln(...))/m, which is sharp in the low-error regime.

## Why it bounds neural nets non-vacuously

Dziugaite & Roy (2017): instead of bounding a *fixed* trained net, **optimize Q directly to minimize the bound** (the RHS is differentiable). Train P-data-independent, then fit Q = N(w, Σ) around the SGD solution by minimizing empirical risk + KL. For MNIST MLPs this gave the first non-vacuous bound (~0.16 error bound) where VC/Rademacher are hopelessly vacuous. Key levers: flat minima → small KL; a good (e.g. data-dependent but admissible) prior shrinks KL.

## Where it appears

- **Dziugaite & Roy (2017)** — optimize the PAC-Bayes bound itself as the training objective; first non-vacuous net bounds
- **Information-theoretic generalization** — PAC-Bayes ≈ mutual-information bounds (Xu–Raginsky, Russo–Zou); KL(Q‖P) ↔ I(W;S)
- **Flat-minima / SAM** — sharpness-aware minimization is morally a PAC-Bayes bound on a perturbed loss 𝔼_{ε}[L(w+ε)]
- **Bayesian deep learning / variational inference** — the KL-to-prior term is structurally identical to the ELBO regularizer

## Common mistake

Believing the prior P can depend on the data used to evaluate the bound. **P must be chosen independently of S** for validity. "Data-dependent priors" are admissible only when fit on a *disjoint* split (or via differential-privacy / sample-splitting arguments) — reusing the training data to pick P and then bounding on the same data breaks the guarantee. Also: the bound is on the *stochastic* Gibbs classifier 𝔼_{h∼Q}[R(h)], not the single deterministic mean predictor (a derandomization step, e.g. margin/majority-vote, is needed).

## See also
- [[kl-divergence]] — the KL(Q‖P) term is the entire complexity penalty
- [[information-bottleneck]] — PAC-Bayes connects to MI-based generalization via I(W;S)
- [[bias-variance]] — an alternative, weaker lens on the same generalization question
