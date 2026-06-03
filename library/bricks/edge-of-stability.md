# Edge of Stability

**One-liner:** During full-batch gradient descent, the top Hessian eigenvalue (sharpness) climbs to ≈2/η and then hovers just above it — training stably reduces loss while sitting at the boundary where classical descent theory says it should diverge.

## The phenomenon (Cohen et al., 2021)

Define sharpness as λ_max(∇²L(θ)), the largest eigenvalue of the loss Hessian. Classical theory: GD with step size η converges only if sharpness < 2/η; above that the quadratic model diverges. Empirically with full-batch GD on neural nets, two phases occur:

1. **Progressive sharpening** — λ_max steadily rises during training.
2. **Edge of stability (EoS)** — once λ_max reaches ≈2/η, it stops rising and oscillates right around 2/η for the rest of training.

At EoS the loss is *non-monotonic on short timescales* (it bounces) yet *decreases over long timescales*. The dynamics are not captured by the descent lemma; the optimizer is self-stabilizing.

```
sharpness ≡ λ_max(∇²L)
stable region (classical):  η · λ_max < 2
observed at EoS:            η · λ_max ≈ 2   (hovers, oscillates)
```

So increasing η doesn't just speed training — it *caps* the sharpness the network is driven to (λ_max ≈ 2/η).

## Where it appears

- Cohen et al. 2021 ("Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability") — the empirical discovery across architectures.
- Sharpness-aware minimization (SAM) and large-LR training — explained as deliberately steering toward / past the sharpness threshold for flatter minima and better generalization.
- LR warmup / schedule design in frontier pretraining — warmup lets sharpness grow before η is large, avoiding the early-training blowup that EoS dynamics predict; the implicit "η selects sharpness" link informs LR tuning and μP transfer.
- Adam analogues ("adaptive EoS") — the threshold becomes ≈38/η on the preconditioned sharpness, relevant to LLM optimizer behavior.

## Common mistake

Assuming EoS means training is broken or about to diverge. The loss-spikes are expected: GD is provably stable in a long-horizon sense even with η·λ_max slightly above 2. Also: EoS is a *full-batch / large-batch deterministic* phenomenon — with small-batch SGD, gradient noise dominates and the clean 2/η ceiling blurs (related but distinct mechanism).

## See also
- [[eigendecomposition]] — sharpness is the top eigenvalue of the loss Hessian
- [[implicit-bias]] — EoS biases GD toward flatter (low-sharpness) solutions
- [[scaling-laws]] — training dynamics at scale connect back to sharpness and stable step size
