# Effective LR & Norm-Growth Dynamics

**One-liner:** Under scale-invariant normalization (LN/RMSNorm) the loss depends only on weight *direction*, so the meaningful step size is the effective LR η_eff ≈ η/‖w‖² which self-corrects toward a rotational equilibrium — making the raw LR alone a misleading knob.

## The key insight

A layer feeding a normalization is scale-invariant: f(cw) = f(w), so the gradient is orthogonal to w (g·w = 0) and homogeneous of degree −1 (‖g(cw)‖ = ‖g(w)‖/c). The directional update is what matters, and its magnitude scales as the **effective learning rate**:

η_eff ≈ η / ‖w‖²

With SGD + weight decay λ, the norm obeys (ignoring the orthogonal gradient's 2nd-order norm growth vs. WD shrinkage):

d‖w‖²/dt ≈ −2λ‖w‖² + η²‖g‖²

This has a fixed point. At **equilibrium** ‖w‖² stops drifting, so η_eff settles to a steady value set jointly by η and λ:

‖w‖*² ∝ √(η/λ) · ‖g‖  ⇒  η_eff* ∝ √(ηλ) / ‖g‖

The product ηλ (the WD-LR interaction), not η, controls the equilibrium step. The weights then trace a near-constant-norm sphere, advancing by **rotation** rather than radial growth — the "rotational equilibrium."

## Where it appears

- Li & Arora (2019), "An Exponential Learning Rate Schedule" — show scale-invariance makes WD ≡ an exponentially-growing LR; raw schedule is not what the network sees.
- Van Laarhoven (2017); Zhang et al. "Three Mechanisms of WD Regularization" — WD's real effect is controlling η_eff, not classical L2 capacity control.
- Kosson et al. (2024), "Rotational Equilibrium" — formalizes per-layer angular update size; motivates per-layer LR normalization (basis for LARS/LAMB-style and Muon-style spectral updates).
- AdamW vs Adam — decoupling WD is exactly so λ cleanly tunes η_eff at equilibrium; coupled L2 entangles it with the second-moment scaling.
- μP / warmup analysis — explains why LR warmup matters: ‖w‖ starts small so early η_eff is huge.

## Common mistake

Reading the configured LR as "the step size." For any sub-network before a norm layer, the network only feels η/‖w‖²; two runs with very different raw η can sit at the same equilibrium η_eff once you account for ‖w‖ and λ. Equivalently, tuning LR while ignoring weight decay (and vice versa) is tuning one composite knob ηλ blindly.

## See also
- [[edge-of-stability]] — effective LR, not raw LR, sets the 2/η sharpness threshold
- [[decoupled-weight-decay]] — AdamW decouples λ precisely to control η_eff
- [[muon-spectral-norm-newton-schulz]] — normalizes update geometry per-layer, sidestepping ‖w‖-dependent effective LR
