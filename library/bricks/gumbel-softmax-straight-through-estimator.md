# Gumbel-Softmax / Straight-Through Estimator

**One-liner:** Two tricks for backprop through discrete sampling — Gumbel-Softmax relaxes a categorical sample into a differentiable softmax over noised logits; Straight-Through (ST) passes the hard one-hot/argmax forward but copies the soft gradient backward, treating the discretization as identity.

## The formula

**Gumbel-Max** (exact discrete sampling): draw gᵢ = −log(−log uᵢ), uᵢ ~ Uniform(0,1); then sample = argmaxᵢ (logitᵢ + gᵢ). The Gumbel noise makes argmax-of-perturbed-logits exactly equivalent to sampling from softmax(logits).

**Gumbel-Softmax** (Jang et al.; Maddison et al. "Concrete", 2017) — replace argmax with temperature-τ softmax:

  yᵢ = exp((logitᵢ + gᵢ)/τ) / Σⱼ exp((logitⱼ + gⱼ)/τ)

τ → 0: y → one-hot (high variance, ~unbiased); τ large: y → uniform (low variance, biased). Anneal τ during training.

**Straight-Through estimator:** forward uses hard = one_hot(argmax(y)); backward uses ∂hard/∂logit ≈ ∂y_soft/∂logit. In code: `hard = onehot(argmax(y)); out = hard + y_soft - y_soft.detach()` (the `.detach()` zeroes the value but keeps soft's gradient).

## Where it appears

- **VQ-VAE / discrete tokenizers** — codebook lookup is an argmax over distances; ST passes the gradient through the non-differentiable nearest-neighbor quantizer to the encoder (the canonical ST use).
- **MoE routing** — top-k expert selection is discrete; some routers use Gumbel noise for exploration / load balancing, ST for differentiable gating.
- **Discrete latents & structured prediction** — Concrete/Gumbel-Softmax for categorical VAEs, hard attention, differentiable subset/permutation sampling.

## Common mistake

Forgetting Gumbel-Softmax is **biased** for τ > 0 — you optimize a relaxed objective, not the true discrete one. And ST is a **biased gradient estimator** (the identity-backward is not the true Jacobian, which is zero a.e.); it works empirically but has no unbiasedness guarantee, unlike REINFORCE/score-function estimators which are unbiased but high-variance.

## See also
- [[reparameterization]] — Gumbel-Softmax is the reparameterization trick adapted to discrete variables
- [[log-derivative-trick]] — REINFORCE, the unbiased high-variance alternative for discrete gradients
- [[vq-vae-discrete-visual-tokenizers]] — the headline application of the straight-through estimator
