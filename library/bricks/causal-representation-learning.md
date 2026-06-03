# Causal Representation Learning (CRL)

**One-liner:** Recover latent causal variables z and their causal graph G from high-dimensional observations x = g(z) (pixels, text), so the learned representation factorizes correctly and stays valid under intervention and distribution shift — not just predicts.

## The setup / definition

Generative model: latent causal variables z = (z₁,…,zₖ) follow an SCM with DAG G, rendered by an unknown (often nonlinear) mixing g:

x = g(z),  zᵢ := fᵢ(z_pa(i), εᵢ)

Goal: from observations of x (across environments/interventions), recover ẑ that equals z up to an **acceptable ambiguity** — i.e. an estimate (ĝ, ẑ, Ĝ) with ẑ = h(z) for a permitted h (permutation + per-node reparam). Plain reconstruction is hopelessly unidentifiable (any invertible h gives ĝ = g∘h⁻¹), so CRL ≠ disentanglement-by-VAE.

## What makes it identifiable

You cannot get z from i.i.d. x alone (Locatello 2019 impossibility). Identifiability needs extra structure:
- **Multiple environments / interventions** — different distributions p^e(z) sharing g; sparse mechanism shifts pin down which zᵢ changed (Ahuja, Brehmer/Locatello "weakly supervised CRL", interventional CRL).
- **Sparsity / mechanism independence** — ICM principle: causal factors vary independently across interventions.
- **Temporal / action structure** — paired (xₜ, xₜ₊₁, aₜ) transitions identify the latent dynamics.

## Where it appears

- **Brehmer/Locatello (weakly-supervised CRL, NeurIPS 2022)** — pre/post-intervention image pairs recover latents up to permutation+rescaling.
- **Interventional/multi-environment CRL (Ahuja, Squires, Varici 2023–24)** — formal identifiability from single-node interventions, even nonparametric.
- **Causal world models / object-centric RL** — agents learning disentangled, intervenable state for OOD generalization and planning.
- **Mechanistic interpretability framing (2024–26)** — SAE features increasingly cast as latent causal variables; "is this a real causal feature?" is a CRL identifiability question.

## Common mistake

Conflating CRL with disentanglement or "interpretable VAE." Statistical independence of latents (β-VAE) is neither necessary nor sufficient for causal recovery — causal factors can be dependent, and independent factors can be entangled mixtures. Without interventions, multiple environments, or a temporal signal, the latents are provably unidentifiable: a good reconstruction loss tells you nothing about whether ẑ are the true causes.

## See also
- [[do-calculus]] — once Ĝ is recovered, do-calculus lets you compute intervention effects on latents
- [[potential-outcomes]] — the counterfactual/intervention semantics CRL aims to make valid in latent space
- [[sparse-autoencoders]] — SAE features as candidate latent causal variables in interpretability
