# LR-Batch-Size Coupling (Linear / Sqrt Scaling)

**One-liner:** To preserve optimization dynamics as you grow the (data-parallel) batch size B, scale the learning rate with B — linearly (η ∝ B) for SGD/momentum, square-root (η ∝ √B) for Adam — but only up to the critical batch size, beyond which the speedup-per-step saturates.

## The formula / definition

Total gradient noise scales as 1/B. To keep the per-step update's signal-to-noise ratio constant when you increase B:

- **Linear rule (SGD, SGD+momentum):** η ∝ B. Intuition: the SGD update is η·(1/B)·Σ gᵢ; doubling B halves variance, so you can double η and keep the noise scale of the parameter trajectory fixed (Goyal et al., 2017, "1 hour ImageNet").
- **Sqrt rule (Adam / adaptive):** η ∝ √B. Adam divides by √(v) ≈ RMS of the gradient, which already partly cancels the magnitude change; the residual variance term scales as √B (Krizhevsky 2014; Malladi et al. 2022 SDE analysis).

**SDE view (Malladi et al., 2022):** SGD ≈ discretization of an SDE with noise ∝ η/B; holding η/B fixed (linear) preserves the continuous trajectory. For Adam the invariant is η/√B.

**Critical batch size B_crit** (McCandlish et al., 2018, gradient noise scale B_noise = tr(Σ)/|g|²): below B_crit, doubling B ≈ halves steps-to-target (perfect scaling); above it, returns diminish and you waste compute. B_crit grows during training as the loss landscape flattens.

## Where it appears

- Goyal et al. "Accurate, Large Minibatch SGD" (2017) — linear scaling + gradual warmup trained ImageNet (batch 8k) in 1 hour; warmup is what makes large-batch start survivable.
- LLM pretraining recipes — Adam at batch 1M–16M tokens use √B-flavored tuning; LR chosen near but below B_crit (gradient-noise-scale estimation guides batch ramp-up schedules in GPT-3/PaLM-era training).
- μP / hyperparameter scaling laws — LR transfer across width is composed with batch-size scaling to set the final η for a target run.
- Data-parallel / FSDP scaling — every time you add GPUs you grow the effective global batch, so η must be re-derived, not held fixed.

## Common mistake

Applying the **linear** rule to **Adam**. Adam is the default LLM optimizer and its correct coupling is closer to √B (or even weaker, with diminishing benefit past B_crit). Linearly scaling Adam's LR with batch size overshoots, causing loss spikes / divergence. The second mistake: assuming any rule holds *past* the critical batch size — beyond B_crit no LR scaling buys you faster convergence, you're just spending compute for nothing.

## See also
- [[critical-batch-size-gradient-noise-scale]] — defines the B_crit ceiling where scaling stops paying off
- [[learning-rate-warmup]] — the prerequisite that makes large-batch (high-η) starts stable
- [[maximal-update-parameterization]] — composes with batch scaling for full HP transfer
