# Reparameterization Trick

**One-liner:** Move the randomness of a sample out of the learnable parameters into fixed external noise — z = μ + σ⊙ε with ε∼N(0,I) — so ∇θ E[f(z)] becomes an expectation of pathwise gradients you can backprop through; the low-variance alternative to the score-function (REINFORCE) estimator.

## The formula / definition

Goal: estimate ∇_φ E_{z∼q_φ(z)}[f(z)] where q_φ depends on parameters φ.

Reparameterize the sample as a deterministic transform of φ and a parameter-free base noise:
  z = g_φ(ε),  ε ∼ p(ε)   (e.g. z = μ_φ + σ_φ ⊙ ε,  ε ∼ N(0,I))

Then the gradient moves inside the expectation (pathwise / "infinitesimal perturbation" estimator):
  ∇_φ E_{q_φ}[f(z)] = E_{p(ε)}[ ∇_φ f(g_φ(ε)) ] = E_{p(ε)}[ f'(z) ∂g_φ/∂φ ]

Contrast — score-function / REINFORCE / log-derivative estimator (no reparam needed):
  ∇_φ E_{q_φ}[f(z)] = E_{q_φ}[ f(z) ∇_φ log q_φ(z) ]

Reparam typically has far lower variance because it uses f's gradient signal, not just its value.

## Where it appears

- VAEs (Kingma & Welling 2013) — the original use: backprop the reconstruction + KL term through the sampled latent z = μ + σ⊙ε.
- Normalizing flows / continuous flows — the whole model *is* an invertible reparameterization z = g_φ(ε); change-of-variables gives exact likelihood.
- Diffusion / score models — DDPM's q(x_t|x_0)=√ᾱ_t x_0 + √(1−ᾱ_t) ε is a reparameterization; the network is trained to predict ε.
- Bayesian deep learning — "local reparameterization" (Kingma 2015) samples weights per-minibatch for lower-variance variational inference; Gumbel-Softmax/Concrete relaxes *discrete* sampling into a differentiable form.
- Policy gradients with continuous actions — reparameterized (pathwise) gradients in SAC vs. score-function in REINFORCE.

## Common mistake

Trying to reparameterize discrete latents directly — there is no differentiable g_φ for a categorical draw, so the trick fails and you fall back to the score-function estimator (with baselines) or a continuous relaxation (Gumbel-Softmax / Concrete). Also: forgetting the trick only applies when q_φ is reparameterizable (location-scale, or invertible-transform families); for general distributions, or when f itself depends on φ, you must add the missing terms.

## See also
- [[log-derivative-trick]] — the higher-variance score-function estimator this replaces
- [[kl-divergence]] — the VAE ELBO term reparameterization lets you optimize end-to-end
- [[ddpm]] — diffusion's forward process is a chained reparameterization in disguise
