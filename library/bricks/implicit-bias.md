# Implicit Bias of SGD

**One-liner:** Among the infinitely many parameter settings that perfectly fit the data, GD/SGD doesn't pick one at random — it converges to structured, low-complexity solutions (min-norm, max-margin, low-rank), and this *unstated* preference is what lets overparameterized nets generalize without explicit regularization.

## The key results

**Linear regression (least squares).** GD from w₀ = 0 converges to the *minimum ℓ₂-norm* interpolator: argmin ‖w‖₂ s.t. Xw = y. The iterates stay in the row space of X; you never move in null-space directions, so you land at the pseudoinverse solution w = X⁺y.

**Separable classification (logistic/exp loss).** GD direction converges to the *ℓ₂ max-margin (hard-margin SVM)* solution, but only *logarithmically slowly*: w(t)/‖w(t)‖ → ŵ_SVM with ‖w(t)‖ ~ log t (Soudry et al. 2018). The loss keeps shrinking long after train accuracy hits 100%.

**Matrix factorization / deep linear nets.** GD on W = U Vᵀ from small init is biased toward *low nuclear norm* (low rank) — an implicit rank regularizer, not a norm penalty (Gunasekar 2017; Arora 2019).

**Geometry depends on the algorithm.** The bias is parameterization- and optimizer-specific:
- GD → ℓ₂ (Euclidean) geometry.
- Mirror descent / steepest descent in a norm ‖·‖ → max-margin in that norm.
- **Adam / sign-based** → closer to ℓ∞ geometry → *different* margin than SGD (this is one reason Adam and SGD generalize differently).

## Stochasticity adds its own bias

SGD ≠ GD. Minibatch noise pushes toward *flat minima*: SGD implicitly minimizes an extra term ∝ (η/4)·‖∇L‖² (squared gradient norm / trace of Hessian), penalizing sharpness. Large learning rate + small batch ⇒ stronger flatness bias, often better generalization. Connects to **edge of stability**, where η sits right at 2/λ_max(Hessian).

## Where it appears

- **Soudry et al. 2018 (max-margin)** — foundational result; explains why test loss improves after zero train error.
- **Double descent / benign overfitting (Belkin, Bartlett 2019-2020)** — min-norm interpolation is *why* overparameterized models overfit harmlessly.
- **Grokking** — delayed generalization is the slow implicit drift from a memorizing interpolator toward the structured (low-norm) one.
- **Why no explicit reg is needed** — Zhang et al. 2017 "rethinking generalization": nets fit random labels yet generalize on real ones; implicit bias is the resolution.

## Common mistake

Thinking it's "just weight decay in disguise." It isn't — the bias arises from the *optimization trajectory and geometry*, not from any term in the loss. The norm being minimized (ℓ₂ vs nuclear vs ℓ∞) is set by the optimizer and parameterization, and depends on initialization scale; explicit regularization is neither necessary nor a faithful proxy for it.

## See also
- [[double-descent]] — min-norm interpolation is the mechanism behind benign overfitting
- [[grokking]] — delayed generalization as slow convergence to the low-norm solution
- [[edge-of-stability]] — SGD's flatness/sharpness bias and the 2/λ_max regime
- [[ntk]] — the lazy-training regime where the implicit bias becomes exactly linear/min-norm
