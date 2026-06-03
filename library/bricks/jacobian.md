# Jacobian

**One-liner:** The matrix of all first-order partials J_{ij} = ∂f_i/∂x_j of a vector map f: ℝⁿ → ℝᵐ; it is the best local linear approximation of f, and the chain rule in vector form is just Jacobian multiplication.

## The formula / definition

For f: ℝⁿ → ℝᵐ, the Jacobian J ∈ ℝ^{m×n}:

J = ∂f/∂x,  J_{ij} = ∂f_i/∂x_j

Local linearization: f(x + δ) ≈ f(x) + Jδ.

Chain rule (composition g∘f): J_{g∘f}(x) = J_g(f(x)) · J_f(x) — Jacobians multiply right-to-left.

Special cases:
- m = 1 (scalar output): J = ∇fᵀ (gradient is a row Jacobian).
- m = n: det(J) is the local volume-scaling factor (change-of-variables).

## The two products autodiff actually computes

Nobody materializes J for large nets (it is m×n, huge). Autodiff gives matrix-free products:
- **JVP** (forward mode): Jv — push a tangent vector forward. Cost ~1 forward pass.
- **VJP** (reverse mode): vᵀJ — pull a cotangent back. Cost ~1 backward pass. This *is* backprop: the loss gradient is a sequence of VJPs through each layer's local Jacobian.

Full J needs n JVPs or m VJPs.

## Where it appears

- **Backpropagation** — each layer contributes its local Jacobian; reverse-mode chains VJPs so you never form the full matrix.
- **Normalizing flows / change-of-variables** — log p(x) = log p(z) + log|det J|; RealNVP/Glow use triangular Jacobians so det is cheap (product of diagonal).
- **Diffusion & flow matching** — divergence ∇·v = tr(J) appears in continuous-time likelihoods (Hutchinson trace estimator: tr(J) ≈ E[εᵀJε] via VJPs).
- **Sensitivity / robustness** — Jacobian spectral norm ‖J‖₂ bounds local Lipschitz constant; adversarial-robustness and stability analyses penalize it.
- **Newton / Gauss-Newton & natural gradient** — JᵀJ approximates curvature; the Fisher/GGN matrix is built from per-example Jacobians.

## Common mistake

Confusing the Jacobian with the Hessian, and confusing JVP with VJP. The Jacobian is first-order (∂f/∂x); the Hessian is second-order (∂²/∂x², the Jacobian *of the gradient*). And direction matters: reverse mode (VJP) is cheap when outputs ≪ inputs (one scalar loss → all params), forward mode (JVP) when inputs ≪ outputs. Backprop is reverse mode precisely because the loss is scalar.

## See also
- [[backpropagation]] — reverse-mode autodiff is the chained VJP of layer Jacobians
- [[ntk]] — the NTK is JJᵀ of the network output w.r.t. parameters
- [[flow-matching]] — likelihoods use the Jacobian trace/determinant of the velocity field
