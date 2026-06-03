# Shampoo / SOAP (Second-Order Preconditioners)

**One-liner:** Adaptive optimizers that precondition each weight matrix with Kronecker-factored second-moment statistics instead of Adam's per-element scaling; SOAP runs Adam inside Shampoo's slowly-updated eigenbasis to recover the curvature gains at near-Adam cost — the benchmark "is second-order worth it" baseline.

## The formula / definition

For a weight matrix W with gradient G (shape m×n), Shampoo maintains two Kronecker factors:
  L += G Gᵀ        (m×m, left)
  R += Gᵀ G        (n×n, right)
Preconditioned update: W -= η · L^{-1/4} · G · R^{-1/4}

This approximates the full-matrix Adagrad preconditioner (G⊗G is mn×mn) by L⊗R, cutting cost from O((mn)²) to O(m²+n²). The −1/4 exponents come from splitting the −1/2 of a single-matrix preconditioner across two factors. Matrix inverse-roots are recomputed every K steps (e.g. K=100) via Newton-Schulz or eigendecomposition; that amortization is what makes it tractable.

**SOAP (Vyas et al., 2024):** diagonalize L and R once to get eigenbases Q_L, Q_R. Rotate the gradient into that basis (G' = Q_Lᵀ G Q_R), run plain **Adam** on the rotated coordinates, then rotate back. The eigenbasis is refreshed every K steps; Adam's second moments inside it are updated every step. This gives Shampoo-quality preconditioning with only one extra hyperparameter (the preconditioning frequency) and far less per-step overhead.

## Where it appears

- **Distributed Shampoo (Anil et al.)** — production-scale impl with blocking, grafting (borrow Adam's step size, Shampoo's direction), and per-layer factor sharding; basis of many "beat Adam" results.
- **AlgoPerf / external optimizer benchmarks (2024)** — Shampoo won the training-algorithms track, putting second-order methods back in serious contention for LLM pretraining.
- **SOAP (2024–2025)** — strongest practical second-order baseline; ~40% fewer steps and ~35% less wall-clock vs AdamW on LM pretraining in the paper's setting.
- **Muon** — the lightweight cousin: orthogonalizes the momentum (spectral-norm view) instead of accumulating Kronecker factors; often the cheaper choice people reach for first.

## Common mistake

Thinking Shampoo is "full second-order / Newton." It is **not** the Hessian — it is a Kronecker-factored approximation to the *Adagrad/Fisher* preconditioner (gradient outer products), and the L⊗R factorization is itself an approximation. Also: forgetting that the inverse-root recompute is amortized over K steps — costing it as if every step does an eigendecomposition wildly overestimates the overhead and is exactly the misconception SOAP exploits.

## See also
- [[muon-optimizer]] — the cheaper spectral-norm sibling; same "structure-aware preconditioning of matrices" idea
- [[fisher-information-natural-gradient]] — Shampoo approximates the Fisher/natural-gradient preconditioner
- [[adam-update-rule]] — the per-element diagonal baseline these methods generalize and are benchmarked against
