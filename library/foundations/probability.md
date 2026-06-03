# Probability & Information Theory

*Calibrated to what frontier papers assume — not a probability course.*

---

## Why this matters at the frontier

- **RLHF and DPO** — the DPO loss is derived by recognizing that the optimal RLHF policy has a specific form in terms of a KL-constrained reward maximization. Without KL divergence cold, you cannot follow this derivation.
- **VAEs and diffusion** — the ELBO in a VAE is a KL term plus a reconstruction term. Score matching in diffusion models estimates the gradient of the log-density. Both require facility with probability distributions and their derivatives.
- **Contrastive learning** — InfoNCE is a lower bound on mutual information. Understanding *why* this objective works requires understanding mutual information.
- **In-context learning theory** — the best theoretical account of ICL frames it as approximate Bayesian inference over a prior on functions. Without Bayes' theorem, this is opaque.
- **Conformal prediction and PPI** — the guarantees are stated in probabilistic language (coverage probability, exchangeability, etc.).

---

## Probability distributions — what matters

A distribution P over X specifies P(X = x) for all x (discrete) or p(x)dx (continuous).

**The key distributions to know cold:**

| Distribution | Where it appears |
|-------------|-----------------|
| Gaussian N(μ, σ²) | Weight initialization, diffusion noise, Bayesian priors |
| Categorical (softmax output) | Next-token prediction, classification |
| Bernoulli | Binary outcomes, preference pairs in RLHF |
| Beta | Bayesian inference on probabilities |
| Dirichlet | Priors over categorical distributions |

**Reparameterization trick:** to sample from N(μ, σ²), sample ε ~ N(0,1) and compute x = μ + σε. This makes sampling differentiable w.r.t. μ and σ — the foundation of VAEs and flow matching.

---

## KL Divergence — know this cold

**KL(P || Q) = Eₓ~P [log P(x)/Q(x)] = Eₓ~P [log P(x)] - Eₓ~P [log Q(x)]**

KL(P||Q) measures how much more you're surprised (in bits/nats) when you model P using Q, compared to modeling P with P itself.

**Properties:**
- KL(P||Q) ≥ 0 (Gibbs' inequality)
- KL(P||Q) = 0 iff P = Q almost everywhere
- **Not symmetric**: KL(P||Q) ≠ KL(Q||P)

**Forward vs. reverse KL:**
- KL(P||Q): "forward KL" — fitting Q to P. Zero-avoiding: Q must cover all of P's support.
- KL(Q||P): "reverse KL" — fitting Q to P differently. Zero-forcing: Q will avoid regions where P is near zero. (VAEs use reverse KL.)

**RLHF connection:** the PPO objective for RLHF is:
```
max E[R(x,y)] - β · KL(π_θ || π_ref)
```
The KL term prevents the policy from deviating too far from the reference policy. The optimal policy has a closed-form expression:
```
π*(y|x) ∝ π_ref(y|x) · exp(R(x,y)/β)
```
This closed-form is what DPO uses — it expresses the reward in terms of the log ratio of the optimal policy to the reference, and eliminates the need for an explicit reward model.

**VAE connection:** the ELBO is -KL(q(z|x) || p(z)) + E[log p(x|z)]. Maximizing ELBO = minimizing KL from approximate posterior to prior + maximizing reconstruction likelihood.

---

## Entropy and information

**Entropy H(X) = -E[log P(X)] = -Σ P(x) log P(x)**

Entropy = average surprise = uncertainty in the distribution.
- Maximum entropy for a discrete distribution with K outcomes: log K (uniform)
- H = 0 for a deterministic distribution

**Cross-entropy H(P, Q) = -E_{X~P}[log Q(X)]**

This is what you minimize during training: the negative log-likelihood of the data (P) under the model (Q).

**Key identity:** H(P, Q) = H(P) + KL(P||Q)

Minimizing cross-entropy = minimizing KL between data distribution and model distribution (since H(P) is fixed).

**Perplexity:** PPL = exp(H(P, Q)) = exp(cross-entropy loss). Perplexity of K means the model is as uncertain as a uniform distribution over K choices. Lower is better.

---

## Mutual Information

**I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = KL(P(X,Y) || P(X)P(Y))**

Mutual information = reduction in uncertainty about X from knowing Y = how much X and Y share.

**InfoNCE lower bound on mutual information:**
```
I(X;Y) ≥ E[log exp(f(x,y)) / Σⱼ exp(f(x,yⱼ))]
```
where the sum is over one positive pair (x,y) and N-1 negative pairs yⱼ. Maximizing this lower bound → the representations f(x) and f(y) capture the shared information.

**Why contrastive learning works:** SimCLR, CLIP, and MoCo all maximize a version of this bound. The temperature τ in the softmax controls how sharp the distribution is — low τ = focus on hardest negatives.

---

## Bayesian inference — the framework behind ICL

**Bayes' theorem:** P(θ|D) = P(D|θ) · P(θ) / P(D)

- P(θ): prior — belief before seeing data
- P(D|θ): likelihood — probability of the data given parameters
- P(θ|D): posterior — updated belief after data
- P(D): marginal likelihood / evidence — normalizing constant

**ICL as Bayesian inference:** the cleanest account of in-context learning (Xie et al., 2021) says: during pretraining, the model learns a prior P(θ) over latent concepts. At inference, the in-context examples D = {(x₁,y₁),...,(xₙ,yₙ)} are treated as evidence, and the model's output approximates the posterior predictive:

```
P(y|x, D) = ∫ P(y|x, θ) P(θ|D) dθ
```

The model *is* a Bayesian predictor, amortized over all possible tasks it was trained on. PFNs make this literal: pretrain on synthetic tasks drawn from P(θ), and inference is exact (approximate) posterior prediction.

---

## Score functions and diffusion

**Score function:** ∇ₓ log p(x) — the gradient of the log-density with respect to the data.

**Score matching** (Hyvärinen, 2005): instead of estimating p(x) directly (requires normalizing constant), estimate the score function directly using:
```
E[‖s_θ(x) - ∇ₓ log p(x)‖²]
```

**Why this matters for diffusion:** DDPM trains a network to predict the noise ε that was added to corrupt data. This is equivalent (with appropriate weighting) to score matching — the denoising network estimates ∇ₓ log p_t(x) at each noise level t. Langevin dynamics then uses this score to sample by gradient ascent on the log-density.

---

## The distributions chain in RLHF/DPO

The full logic chain (crucial for reading DPO):

1. **RLHF objective:** max_π E_{x~D, y~π}[R(x,y)] - β·KL(π||π_ref)
2. **Optimal policy (analytic solution):** π*(y|x) ∝ π_ref(y|x)·exp(R(x,y)/β)
3. **Rearranging for R:** R(x,y) = β·log[π*(y|x)/π_ref(y|x)] + β·log Z(x)
4. **Bradley-Terry preference model:** P(y_w ≻ y_l) = σ(R(x,y_w) - R(x,y_l))
5. **Substitute step 3 into step 4** → R cancels Z(x)
6. **DPO loss:** -log σ(β·[log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)])

Without knowing KL divergence and the concept of an optimal policy under a KL constraint, steps 1–3 are incomprehensible. With them, DPO is a simple substitution.
