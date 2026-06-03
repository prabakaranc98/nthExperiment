# Sharpness-Aware Minimization (SAM) & Flat Minima

**One-liner:** Flat minima (low loss curvature in a neighborhood) tend to generalize better; SAM optimizes the worst-case loss in an ε-ball around the weights via a gradient-ascent step to find the local maximizer, then a descent step there — biasing training toward flat solutions at ~2x the gradient cost.

## The formula / definition

SAM minimizes the **sharpness-aware objective** — the max loss within an ℓ2 ball of radius ρ:

  min_w  max_{‖ε‖₂ ≤ ρ}  L(w + ε)

The inner max is solved by a first-order (linearized) approximation. The worst-case perturbation points along the gradient:

  ε̂(w) = ρ · ∇L(w) / ‖∇L(w)‖₂

Then take the descent step using the gradient evaluated at the perturbed point:

  w ← w − η · ∇L(w)|_{w + ε̂(w)}

Two forward/backward passes per step: pass 1 computes ε̂ (the ascent direction), pass 2 computes the actual update gradient at w + ε̂. Cost ≈ 2x. The (dropped) higher-order term means SAM's gradient ≈ ∇L(w) + ρ · (∇²L · ∇L)/‖∇L‖, so it penalizes the loss-aligned Hessian curvature.

**Flatness measure:** a minimum is "flat" if the top Hessian eigenvalues λ_max(∇²L) are small / the loss is nearly constant in a neighborhood. SAM implicitly regularizes λ_max.

## Where it appears

- **SAM (Foret et al., 2021)** — SOTA-at-the-time gains on CIFAR/ImageNet; especially strong with limited data and label noise, where flat solutions resist memorization.
- **ASAM (Adaptive SAM)** — scale-invariant ρ so the ε-ball adapts to per-weight magnitude (fixes that raw SAM is sensitive to weight reparameterization).
- **GSAM, SAF, LookSAM, ESAM** — variants that reduce the 2x overhead (e.g., periodic/sparse ascent steps, surrogate gap).
- **ViT / MLP-Mixer training** — SAM substantially closes the gap with conv-net inductive bias by smoothing the loss landscape (Chen et al., 2022); used to make data-hungry architectures trainable without huge data.
- **Bigger picture:** the flat-minima hypothesis underlies why large-batch training (sharp minima) can generalize worse, and connects to PAC-Bayes generalization bounds.

## Common mistake

Conflating "flatness" with guaranteed generalization, and treating sharpness as a coordinate-free quantity. Sharpness is **not reparameterization-invariant** (Dinh et al., 2017): you can rescale weights to make any minimum arbitrarily sharp or flat while leaving the function — and its generalization — unchanged. Naive SAM inherits this; ASAM-style adaptive radii exist precisely to address it. Also: SAM's benefit is largely an *implicit-bias / optimization-trajectory* effect, not purely "the final minimum is flat."

## See also
- [[edge-of-stability]] — training naturally hovers at λ_max ≈ 2/η; SAM actively drives λ_max down
- [[pac-bayes]] — the generalization bound that gives flatness a theoretical justification
- [[implicit-bias]] — SAM is an explicit version of the flatness-seeking bias of SGD
