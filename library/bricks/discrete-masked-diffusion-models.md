# Discrete / Masked Diffusion Models

**One-liner:** Diffusion over discrete tokens via an absorbing-state (`[MASK]`) forward process or general continuous-time Markov chains, trained by score-entropy / masked-token ELBO and decoded by parallel, any-order unmasking — the basis of SEDD, MDLM, and 2025 diffusion LLMs.

## The formula / definition

Two equivalent framings of the forward corruption on a vocab of size V:

- **CTMC / score-based (SEDD):** tokens evolve by a rate matrix Q_t; the reverse process needs the *concrete score* ratios p_t(y)/p_t(x). Train with **score entropy** (a denoising-score-matching loss for discrete states), not L2.
- **Absorbing / masked (MDLM, D3PM-absorbing):** each token independently jumps to `[MASK]` with prob 1−α_t, α_0=1, α_1=0. Reverse step un-masks.

MDLM reduces to a clean, weighted masked-LM ELBO:

L = E_t E_{x_t} [ (α_t' / (1−α_t)) · Σ_{i: x_t^i = MASK} log p_θ(x_0^i | x_t) ]

i.e. a **time-weighted cross-entropy over the currently-masked positions only**, with a model that never re-masks (zero-masking / "carry-over" parameterization). At inference, repeatedly predict all masked tokens, commit a subset (by confidence or schedule), feed back, iterate over T ≪ length steps.

## Where it appears

- **SEDD** (Lou, Meng, Ermon 2024) — score-entropy discrete diffusion; first to match/beat GPT-2 perplexity bounds with a non-AR model.
- **MDLM** (Sahoo et al. 2024) / **MD4** — simplified masked-diffusion ELBO; showed absorbing-state ≈ "any-order BERT" and is the dominant recipe.
- **LLaDA / Dream-7B (2025)** — scaled diffusion LLMs (~7B) competitive with AR baselines; full parallel/any-order decoding.
- **Gemini Diffusion / Mercury (Inception, 2025)** — commercial diffusion LLMs marketed for high-throughput parallel decode.
- **Block diffusion (BD3-LM)** — AR over blocks, diffusion within, interpolating AR and pure diffusion; enables KV-cache + variable length.

## Common mistake

Thinking masked diffusion gives a *cheaper* likelihood or strictly fewer FLOPs than AR. The ELBO is a (often loose) *bound* on log-likelihood — masked-diffusion perplexity numbers are upper bounds, not directly comparable to exact AR perplexity. The real win is parallel/any-order decoding and bidirectional context, not log-likelihood; and naive few-step decoding trades quality for speed because committing many tokens at once ignores their joint dependence.

## See also
- [[masked-parallel-token-generation]] — the decoding mechanism (parallel unmasking) these models use
- [[score-based-models-sdes]] — continuous-state analog; SEDD ports the score-matching idea to discrete CTMCs
- [[diffusion-forcing-block-autoregressive-diffusion]] — block-AR + diffusion hybrid that bridges to these discrete variants
