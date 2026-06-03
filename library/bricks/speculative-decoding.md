# Speculative Decoding

**One-liner:** A cheap draft model proposes k tokens autoregressively; the expensive target model verifies all k in one parallel forward pass and accepts a prefix via a rejection-sampling rule — strictly faster decoding with output distribution provably identical to sampling from the target alone.

## The algorithm (Leviathan et al. / Chen et al., 2023)

Draft model q proposes tokens x₁..x_k. Target model p scores them in a single batched pass (one forward over all k positions, exploiting that decode is memory-bound, not compute-bound). Walk left to right; for draft token xᵢ:

- Accept with probability min(1, p(xᵢ)/q(xᵢ)).
- On first rejection, resample from the residual distribution: x ~ normalize(max(0, p(·) − q(·))), then stop.
- If all k accepted, sample one extra "free" token from p.

This is rejection sampling: the accept rule + residual resample guarantee the kept tokens are **exact samples from p**. Expected tokens per target call = (1 − α^{k+1})/(1 − α), where α is the per-token acceptance rate (alignment between q and p).

## Where it appears

- **Original speculative sampling** (Leviathan 2023; Chen/DeepMind 2023) — small model drafts, large model verifies; 2–3× latency reduction at unchanged quality.
- **Medusa / EAGLE / EAGLE-2/3** — replace the separate draft model with extra heads or a lightweight feature-level autoregressive head on the target itself; EAGLE drafts in feature space and is SOTA in production (2024–2026).
- **Self-speculative / Lookahead / n-gram drafting** — draft from the model's own early layers, or from a prompt-conditioned n-gram cache (no draft model at all).
- **vLLM, TensorRT-LLM, SGLang** — all ship spec-decode; tree/Medusa-style verification of multiple candidate continuations per step.

## Common mistake

Believing greedy/temperature-0 decoding is required, or that it is "approximate/lossy." With the rejection rule it is **distribution-exact** for any temperature — output is statistically indistinguishable from vanilla target sampling. The separate failure mode: it only helps when decode is memory-bandwidth-bound (small batch). At large batch the target pass is already compute-bound, acceptance gains shrink, and the draft overhead can make it *slower*.

## See also
- [[kv-cache]] — verification reuses/extends the target KV cache; rejected drafts roll it back
- [[inference-and-serving]] — spec decoding is a core latency-reduction lever in serving stacks
- [[kl-divergence]] — acceptance rate α tracks how close the draft q is to the target p
