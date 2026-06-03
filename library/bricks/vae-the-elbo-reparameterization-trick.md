# VAE & the ELBO / Reparameterization Trick

**One-liner:** Amortized variational inference that maximizes an evidence lower bound on log p(x), using the reparameterization trick to push low-variance Monte Carlo gradients through a sampled latent — and the perceptual autoencoder that latent diffusion runs on.

## The objective

We want log p(x) = log ∫ p(x|z)p(z) dz but the integral is intractable. Introduce an encoder q_φ(z|x) and bound it:

log p(x) ≥ ELBO = E_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z))

- Term 1 = reconstruction (decoder fits x given z)
- Term 2 = KL pulls the posterior toward the prior p(z) = N(0, I)
- The gap log p(x) − ELBO = D_KL(q_φ(z|x) ‖ p_θ(z|x)) ≥ 0, so tight bound ⇔ accurate posterior

"Amortized": one network q_φ predicts (μ(x), σ(x)) for all x, instead of optimizing per-datapoint variational params.

## The reparameterization trick

You cannot backprop through `z ~ N(μ, σ²)` — sampling is not differentiable. Rewrite the sample as a deterministic function of params plus parameter-free noise:

z = μ_φ(x) + σ_φ(x) ⊙ ε,   ε ~ N(0, I)

Now ∇_φ E[f(z)] = E_ε[∇_φ f(z)], a low-variance pathwise gradient. With Gaussian q and p, the KL term is closed-form: ½ Σ (μ² + σ² − log σ² − 1).

## Where it appears

- **Latent Diffusion / Stable Diffusion, SDXL, SD3** — a KL- or VQ-regularized VAE compresses pixels to a latent (e.g. 8x downsample); diffusion runs entirely in that latent space
- **VQ-VAE / dVAE → DALL·E, MagViT, modern visual tokenizers** — discrete-latent variant swapping the reparameterization trick for codebook lookup + straight-through estimator
- **VampPrior, IWAE, β-VAE, NVAE/VDVAE** — tighter bounds, learned priors, disentanglement knobs, deep hierarchical latents

## Common mistake

Confusing the ELBO's KL term with the reparameterization trick — they are orthogonal. The trick is just a gradient estimator for the reconstruction expectation; the KL is usually closed-form and needs no sampling. Also: posterior collapse — with a strong autoregressive decoder the model ignores z, driving the KL to 0; mitigated by KL warmup / free bits / β scheduling.

## See also
- [[latent-diffusion]] — runs diffusion in the VAE latent; the VAE is its compression backbone
- [[reparameterization]] — the general pathwise gradient estimator this brick instantiates
- [[vq-vae-discrete-visual-tokenizers]] — discrete-latent cousin used as the tokenizer for autoregressive image models
