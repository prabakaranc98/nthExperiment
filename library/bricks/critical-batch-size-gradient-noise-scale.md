# Critical Batch Size & Gradient Noise Scale

**One-liner:** The largest batch that still yields near-linear speedup in steps-to-target-loss, predictable from the gradient noise scale (signal-to-noise ratio of the gradient); it grows during training, motivating batch-size warmup and joint LR–batch tuning.

## The definition (McCandlish et al., 2018)

The simple gradient noise scale is the ratio of gradient variance to gradient magnitude:

B_simple = tr(Σ) / |G|²

where G = E[g] is the true gradient and Σ = Cov(g) is the per-example gradient covariance. Estimate both from gradients at two batch sizes (B_small, B_big):

|G|² ≈ (B_big·|G_big|² − B_small·|G_small|²) / (B_big − B_small)
tr(Σ) ≈ (|G_small|² − |G_big|²) / (1/B_small − 1/B_big)

**Critical batch size B_crit ≈ B_noise.** Below it: ~linear reduction in steps per added example (compute-cheap, time-cheap). Far above it: diminishing returns — you spend more compute for negligible step savings.

## The tradeoff curve

Steps to target loss and examples processed trade off via a Pareto hyperbola:

(S/S_min − 1)(E/E_min − 1) = 1

S_min = serial steps at infinite batch; E_min = total examples at tiny batch. The "knee" sits at B_crit, where S = 2·S_min and E = 2·E_min — the efficient operating point.

## Why it grows during training

As loss decreases, |G|² shrinks faster than gradient noise tr(Σ), so B_noise rises (often by 10–100x from start to end). Early steps tolerate only small batches; late training can absorb huge ones.

## Where it appears

- **OpenAI "An Empirical Model of Large-Batch Training" (McCandlish et al., 2018)** — defines B_noise, predicts B_crit across vision/RL/LM tasks
- **Batch-size warmup / ramp** — GPT-3, PaLM, and many frontier runs ramp batch size up over training to track rising B_crit, saving wall-clock without wasting compute
- **LR–batch coupling** — linear scaling rule (LR ∝ B) and √B (Adam) hold only for B ≪ B_crit; past it you must re-tune
- **Data-parallel scaling decisions** — B_crit caps useful DP width before you must add gradient accumulation or other parallelism

## Common mistake

Treating critical batch size as a fixed property of the model. It is a property of the current loss landscape and grows over training; a batch that is wasteful at step 0 can be efficient near convergence. Also: B_crit is set by gradient SNR, not by GPU memory — fitting a batch in memory says nothing about whether it is past the knee.

## See also
- [[lr-batch-size-coupling]] — linear/√B scaling rules valid only below B_crit
- [[gradient-accumulation-micro-batching]] — how you actually reach a target batch size above per-device limits
- [[loss-spikes-training-instability]] — over-large batches at wrong LR trigger instability
