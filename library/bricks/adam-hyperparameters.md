# Adam Hyperparameters (betas, epsilon, bias correction)

**One-liner:** β1 (momentum decay), β2 (second-moment decay), and ε (denominator floor) set Adam's per-coordinate adaptive step, with the 1−βᵗ bias correction undoing the zero-initialization bias of the EMAs — and large β2 plus mis-scaled ε are recurring causes of LLM training instability and irreproducibility.

## The formula / definition

EMAs of gradient and squared gradient, then bias-corrected:

```
m_t = β1·m_{t-1} + (1−β1)·g_t          # 1st moment (momentum)
v_t = β2·v_{t-1} + (1−β2)·g_t²         # 2nd moment (uncentered variance)
m̂_t = m_t / (1−β1^t)                   # bias correction
v̂_t = v_t / (1−β2^t)
θ_t = θ_{t-1} − η · m̂_t / (√v̂_t + ε)
```

Defaults: β1=0.9, β2=0.999, ε=1e-8. **Bias correction** matters because m,v start at 0, so early estimates are biased toward 0; the 1/(1−βᵗ) factors inflate them, acting as an implicit LR warmup that decays as t grows.

## Where it appears

- **LLM pretraining** — GPT-3/PaLM/LLaMA use β2≈0.95 (not 0.999) for faster adaptation and stability on noisy large-batch gradients; ε often 1e-8 or smaller.
- **Transformer instability** — loss spikes correlate with stale/small v̂ when β2 is too high; the √v̂+ε denominator collapsing or a too-large ε both flatten the per-coordinate scaling.
- **bf16 training** — ε placement matters: `√(v̂)+ε` (PyTorch) vs `√(v̂+ε)` (older TF) differ; small ε in low precision can underflow.

## Common mistake

Treating ε as a tiny numerical guard with no effect. It is a regularizer on the step size: in low-gradient regions √v̂ ≈ 0, so ε caps the maximum step (≈ η/ε scaling of m̂). Changing ε by orders of magnitude silently changes the effective LR and is a top repro gap. Likewise, β2 is not "just momentum for the variance" — it sets the timescale (≈1/(1−β2) steps) over which curvature estimates are averaged.

## See also
- [[adam-update-rule]] — the full Adam algorithm these knobs parameterize
- [[adamw]] — decouples weight decay from this adaptive denominator
- [[loss-spikes-training-instability]] — β2/ε are common levers when spikes appear
