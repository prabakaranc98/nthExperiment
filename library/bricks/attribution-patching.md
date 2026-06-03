# Attribution Patching

**One-liner:** A first-order Taylor approximation of activation patching that estimates the causal effect of swapping every component (or edge) from clean to corrupt activations using a single forward + backward pass, turning O(#components) patching runs into O(1) and scaling causal localization to whole models.

## The formula / definition

Activation patching measures the metric change when an activation a is replaced by its corrupted value a_corrupt: ΔL = L(a ← a_corrupt) − L(a_clean). Attribution patching linearizes this around the clean run:

  ΔL ≈ (a_corrupt − a_clean) · ∂L/∂a

where ∂L/∂a is the gradient of the metric w.r.t. activation a, taken on the **clean** forward pass. You cache clean activations + clean gradients and corrupt activations from two runs, then take the elementwise dot product per component — every node's effect comes out of one backward pass.

**Edge Attribution Patching (EAP):** approximate the effect on the edge u→v of patching u's contribution into v's input:

  effect(u→v) ≈ (a_u^corrupt − a_u^clean) · ∂L/∂(input to v)

This scores all edges in the computational graph simultaneously, giving an approximate attention-and-MLP-edge importance map for circuit discovery. EAP-IG adds an integrated-gradients path to fix zero-gradient/saturation failures.

## Where it appears

- **EAP / EAP-IG (Syed et al. 2023; Hanna et al. 2024)** — automated circuit discovery; replaces ACDC's iterative patch-and-prune (thousands of forward passes) with one gradient pass over all edges, then threshold.
- **ACDC comparison** — attribution patching is the fast first-pass approximation; ACDC remains the slower exact-ish baseline for finding sparse circuits.
- **Attribution graphs / cross-layer transcoders (Anthropic 2025)** — gradient-based attribution scales feature-level circuit tracing on production models.
- **Localization for editing/steering** — quickly ranking heads/MLPs/SAE-features before doing exact patching on the survivors.

## Common mistake

Trusting the linear approximation where it breaks: it is only accurate for small clean→corrupt activation differences. Near saturated nonlinearities (softmax, LayerNorm, attention that flips which token it attends to) the gradient is ~0 or wildly off, so attribution patching *underestimates or misses* high-effect components. Always validate top-ranked nodes with exact activation patching; use integrated-gradients variants when gradients saturate.

## See also
- [[activation-patching-causal-tracing]] — the exact (non-linearized) intervention attribution patching approximates
- [[circuits]] — the structures EAP-based discovery is trying to recover
- [[transcoders-attribution-graphs]] — feature-level attribution graphs built on the same gradient idea
