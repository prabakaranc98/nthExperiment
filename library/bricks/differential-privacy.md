# Differential Privacy (DP-SGD)

**One-liner:** Per-example gradient clipping + Gaussian noise on the gradient sum, giving each training example a formal (ε,δ) bound on how much any output can depend on it — the standard primitive for provably private training.

## The definition

A randomized mechanism M is (ε,δ)-DP if for all adjacent datasets D, D′ (differing by one example) and all outputs S:

P(M(D) ∈ S) ≤ e^ε · P(M(D′) ∈ S) + δ

Smaller ε = stronger privacy; δ is the failure-probability slack (typically δ ≈ 1/|D| or smaller). ε is a *per-mechanism* budget that composes across releases.

## DP-SGD (Abadi et al., 2016)

Per step, on a Poisson-sampled minibatch:

1. Compute **per-example** gradients g_i (not the batch-averaged gradient).
2. **Clip** each to L2 norm C:  ḡ_i = g_i / max(1, ‖g_i‖₂ / C)
3. **Add noise** to the sum:  g̃ = (1/B)(Σ_i ḡ_i + N(0, σ²C²I))
4. Step with g̃ on any optimizer.

Clipping bounds one example's sensitivity to C; noise scale σ + sampling rate q = B/|D| + step count T determine ε. Tight accounting via the **moments accountant / Rényi-DP / PRV accountant** (much tighter than naive (ε,δ) composition).

## Where it appears

- **DP fine-tuning of LLMs** — Yu et al. / Li et al. (2022): LoRA/full-tune with DP-SGD; large pretrained models tolerate DP noise far better than from-scratch training.
- **Apple / Google production** — federated learning with DP for keyboard, telemetry; ghost-clipping / functorch vmap make per-example grads tractable.
- **Membership-inference / extraction defenses** — DP is the formal guarantee that bounds MIA advantage and verbatim memorization.
- **PATE** — alternative DP route: noisy teacher-ensemble voting to label public data.

## Common mistake

Reporting ε without δ, the clipping norm C, the sampling rate, the accountant used, and **what "one example" means** (a row? a user? a document?). The unit of privacy (example-level vs user-level) changes the guarantee entirely, and a large ε (e.g. ε=8) gives a much weaker bound than its single number suggests.

## See also
- [[membership-inference-training-data-extraction]] — the concrete attack DP is designed to provably bound
- [[gradient-clipping]] — same clip op, but per-example and for sensitivity, not stability
- [[machine-unlearning]] — the alternative goal of removing one example's influence after training
