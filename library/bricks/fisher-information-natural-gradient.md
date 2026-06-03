# Fisher Information & Natural Gradient

**One-liner:** The Fisher F is the curvature of the log-likelihood (expected outer product of score, = expected Hessian of NLL); preconditioning the gradient by F⁻¹ gives the natural gradient — the steepest-descent direction in the KL/Riemannian geometry of the model, invariant to reparameterization.

## The formula / definition

Score: s_θ = ∇_θ log p_θ(x). Fisher Information Matrix:

F(θ) = E_{x∼p_θ}[ ∇_θ log p_θ(x) ∇_θ log p_θ(x)ᵀ ] = −E[ ∇²_θ log p_θ(x) ]

(the two forms are equal under regularity). F is the **local 2nd-order Taylor term of KL**: KL(p_θ ‖ p_{θ+δ}) ≈ ½ δᵀ F δ.

**Natural gradient** (Amari 1998): θ ← θ − η F⁻¹ ∇_θ L. This is steepest descent under the KL metric, hence reparameterization-invariant — the update direction doesn't depend on how you coordinatize θ.

Key subtlety: F is an expectation over **model** samples y∼p_θ(·|x) ("true Fisher"). Using the empirical labels instead gives the **empirical Fisher**, which is ≠ Fisher and not the Gauss-Newton matrix (a common source of bugs).

## Where it appears

- **K-FAC / Shampoo / SOAP** — approximate F⁻¹ as Kronecker factors per layer (F ≈ A ⊗ G); the practical way to run natural gradient at scale. Gauss-Newton ≈ Fisher for exp-family losses.
- **EWC (Kirkpatrick 2017, continual learning)** — diagonal of F weights an L2 penalty (F_i (θ_i − θ*_i)²) anchoring high-curvature (important) params to slow forgetting.
- **TRPO / natural policy gradient** — F is the FIM of the policy; the KL-trust-region constraint is exactly ½δᵀFδ ≤ ε, solved via conjugate gradient (Adam's diagonal preconditioner is a crude empirical-Fisher cousin).
- **muP / spectral conditioning intuitions** — the "right" per-layer LR is the one that respects the loss geometry; Fisher is the formal object behind why naive SGD is ill-conditioned across widths.

## Common mistake

Using the **empirical Fisher** (outer product of gradients on observed labels) and calling it the Fisher. Near a minimum or with wrong labels it does NOT approximate the Hessian or Gauss-Newton matrix — it can be badly biased. The true Fisher requires sampling y from the model's predictive distribution, not from the data.

## See also
- [[catastrophic-forgetting-continual-learning]] — EWC uses diagonal Fisher as the importance prior
- [[shampoo-soap]] — second-order optimizers that approximate F⁻¹ via Kronecker factors
- [[ppo-clipped-surrogate-objective]] — PPO is the cheap first-order replacement for TRPO's exact natural-gradient/KL trust region
