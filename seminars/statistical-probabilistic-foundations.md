# Statistical & Probabilistic Inference, Learning, Modeling & Reasoning

### Foundations for Frontier AI

*Frontier AI is applied probability and statistics, executed at scale.*

**FAIRE context:** the P0 foundation under everything — read alongside the [Curriculum Index](curriculum-index.md).

Strip away the engineering vocabulary — pretraining, RLHF, diffusion, evals, calibration — and frontier AI is a small set of probabilistic operations run at enormous scale. Pretraining is maximum likelihood estimation. RLHF, diffusion sampling, and in-context learning are forms of inference. Benchmarks are statistical experiments, and the numbers they produce are estimates with sampling distributions. This document is the conceptual spine for that view: the probability that is the language, the inference that turns data into claims, the modeling that writes distributions over structured worlds, the learning theory that asks when any of it generalizes, the decision theory that turns belief into action — and the frontier systems that are all of the above wearing different names.

> **The one idea.** A frontier model is a parameterized distribution `p_θ` over a structured object (token sequences, pixels, trajectories). *Training fits it (maximum likelihood), generation samples it (inference), alignment constrains it (KL-regularized inference), reasoning conditions it (Bayesian updating), and evaluation measures it (a statistical experiment).* Everything else is a special case — and the only real question is whether you account for the uncertainty explicitly or let it surprise you in production.

| Section | What you get |
|---|---|
| [Probability — the language](#probability--the-language) | The notation deep learning is written in: distributions, expectation, Bayes, exchangeability, the inequalities that bound behavior, information theory |
| [Statistical inference — from data to claims](#statistical-inference--from-data-to-claims) | How a finite sample licenses a population claim: estimation, frequentist vs Bayesian, UQ, testing/power, and the distribution-free toolkit (conformal, PPI, e-values) |
| [Probabilistic modeling — distributions over structured worlds](#probabilistic-modeling--distributions-over-structured-worlds) | The grammar for writing `p_θ`: generative vs discriminative, latent variables and the ELBO, graphical models, the generative zoo, and the diffusion/score/flow unification |
| [Learning theory — generalization as a statistical question](#learning-theory--generalization-as-a-statistical-question) | Why overparameterized models generalize: ERM, double descent, PAC-Bayes, NTK vs feature learning, implicit bias, scaling laws, emergence |
| [Probabilistic reasoning & decision-making](#probabilistic-reasoning--decision-making) | Turning belief into action: sequential Bayes, expected utility, exploration, experimental design, causality, and ICL as implicit Bayes |
| [Where it all shows up at the frontier](#where-it-all-shows-up-at-the-frontier) | The map from foundations to 2024–2026 systems: pretraining=MLE, alignment=KL-inference, generation=score/flow, ICL=amortized Bayes, evals=experiments, UQ=the contract you ship |

---

## Probability — the language

Every claim in modern AI — that a model "learned" a distribution, that a policy "improved," that an eval "measures" capability — is a probabilistic statement. Probability is not a topic adjacent to deep learning; it is the notation in which deep learning is written, and fluency in it is the difference between operating the machinery and understanding it.

### Random variables, distributions, and the two regimes

A random variable `X` is a measurable function from outcomes to values; a *distribution* `p(x)` assigns mass/density to those values. In frontier ML two regimes recur:

- **Discrete, high-dimensional, autoregressive:** `p(x_{1:T}) = ∏_t p(x_t | x_{<t})` — the entire LLM pretraining objective lives here. The "distribution" is over token sequences; the model is a parametric `p_θ`.
- **Continuous, latent:** Gaussians in VAEs/diffusion, where `p(x) = ∫ p(x|z)p(z) dz` is intractable and must be approximated.

The central object is almost never a closed-form `p`; it is an *implicitly defined* `p_θ` we can sample from or score, which is why everything downstream becomes an inference or estimation problem.

### Expectation and variance

Expectation `E[X] = ∫ x p(x) dx` is the linear functional under which losses, returns, and gradients are all defined. Two facts do enormous work:

- **Linearity:** `E[aX + bY] = aE[X] + bE[Y]` regardless of dependence — the reason minibatch gradients are unbiased estimates of the full gradient.
- **Variance** `Var(X) = E[X²] − E[X]²` controls estimator quality. The entire RL-for-LLMs literature (PPO, GRPO, REINFORCE-with-baselines) is variance reduction of a Monte Carlo estimate of `∇_θ E_{π_θ}[R]`; baselines subtract a control variate that leaves the expectation unchanged but shrinks variance.

### Conditional probability and Bayes

`p(x,y) = p(y|x)p(x)`, and Bayes inverts the conditional:

```
p(θ | D) = p(D | θ) p(θ) / p(D)
```

- **Posterior = likelihood × prior, normalized.** Pretraining maximizes the likelihood term; the prior is implicit (architecture, init, regularization).
- **In-context learning** is increasingly read as *implicit Bayesian inference*: the prompt is evidence `D`, and the model's next-token distribution approximates a posterior predictive `p(x_{new} | D)` over latent tasks — a framing developed in full below that explains why more/cleaner demonstrations sharpen behavior without weight updates.
- The intractable denominator `p(D)` (the *evidence*) is the wall that forces variational and Monte Carlo methods everywhere.

### Independence and exchangeability

- **Independence:** `p(x,y) = p(x)p(y)`. The i.i.d. assumption underwrites the generalization theory we mostly violate at frontier scale (web data is neither independent nor identically distributed).
- **Exchangeability** (joint invariant to permutation) is the *weaker, more honest* assumption — by de Finetti, an exchangeable sequence is i.i.d. conditional on a latent parameter, which is exactly the ICL-as-Bayes picture above. It is also the sole assumption behind conformal prediction's finite-sample coverage. Permutation-invariant set architectures and attention's symmetry are exchangeability made architectural.

### Inequalities that bound behavior

| Inequality | Statement | Where it bites |
|---|---|---|
| **Jensen** | `f(E[X]) ≤ E[f(X)]` for convex `f` | The ELBO: `log p(x) ≥ E_q[log p(x,z)/q(z)]` — VAEs/diffusion train the bound, not the target |
| **Hoeffding** | `P(\|X̄ − E[X]\| ≥ t) ≤ 2exp(−2nt²/R²)` for bounded vars | Eval confidence intervals: how many benchmark items to trust a win-rate to ±t |
| **Markov/Chebyshev** | tail mass from moments | Crude but assumption-light bounds on rare-event/safety probabilities |

Concentration is *why* finite evals say anything about a distribution at all; mid-2026 eval hygiene (reporting CIs on win-rates, bootstrapping LLM-judge scores) is Hoeffding/CLT applied honestly.

### Information theory

The currency of "how much a distribution knows."

- **Entropy** `H(p) = −Σ p(x) log p(x)` — irreducible uncertainty; the floor on lossless compression. Sampling temperature is direct entropy control.
- **Cross-entropy** `H(p,q) = −Σ p(x) log q(x)` — *the* pretraining loss. Minimizing `H(p_data, q_θ)` over data is maximum likelihood; the units (nats/bits per token) are literal compression rates, which is why "loss" and "bits-per-byte" are interchangeable in scaling-law plots.
- **KL divergence** `D_KL(p‖q) = Σ p(x) log p(x)/q(x) = H(p,q) − H(p)` — non-negative, asymmetric, not a metric. It is the regularizer of the modern stack:
  - **RLHF/RLAIF:** the reward is penalized by `β·D_KL(π_θ ‖ π_ref)` to keep the policy near the SFT model; DPO/IPO bake this KL constraint into a closed-form loss, eliminating the explicit reward model.
  - **Variational inference:** minimizing `D_KL(q ‖ p(·|x))` *is* maximizing the ELBO — the two are the same move seen from opposite ends.
  - Forward vs. reverse KL is the mode-covering vs. mode-seeking distinction that decides whether a fit hedges or commits.
- **Mutual information** `I(X;Y) = D_KL(p(x,y)‖p(x)p(y)) = H(X) − H(X|Y)` — dependence measured in bits. It is the conceptual core of self-supervised representation learning: InfoNCE (the contrastive loss in CLIP-style and audio/JEPA encoders) is a tractable lower bound on `I` between views. The InfoMax view says a good representation is one that shares maximal MI with the signal and minimal MI with nuisance — the same ledger that the information bottleneck `min I(X;Z) − β I(Z;Y)` writes explicitly.

Held together: **pretraining is cross-entropy minimization (MLE), alignment is KL-constrained policy optimization, representation learning is MI maximization, and evaluation is concentration applied to samples.** The rest of this document is these four sentences, expanded.

## Statistical inference — from data to claims

Probability tells you how a distribution behaves; *inference* runs the arrow backward — from a finite sample to a statement about the distribution that produced it. Every frontier-AI claim — "this checkpoint is better," "the model is calibrated," "RLHF reduced harmful outputs by 12%" — is exactly that inferential leap. Inference is the discipline that quantifies whether the leap is warranted; without it, leaderboards and ablations are anecdotes.

### Estimation: from likelihood to posterior

Given data $D$ and a parametric model $p_\theta$, the three canonical estimators differ in what they target:

- **MLE:** $\hat\theta_{\text{MLE}} = \arg\max_\theta \log p_\theta(D)$. Pretraining *is* MLE — next-token cross-entropy is the negative log-likelihood of an autoregressive factorization $\log p_\theta(x) = \sum_t \log p_\theta(x_t \mid x_{<t})$. Asymptotically efficient and consistent under regularity; achieves the Cramér–Rao bound $\operatorname{Var}(\hat\theta) \ge I(\theta)^{-1}$ where $I$ is Fisher information.
- **MAP:** $\arg\max_\theta \log p_\theta(D) + \log p(\theta)$ — MLE plus a prior. Weight decay is a Gaussian prior; the prior is a regularizer that vanishes as $n\to\infty$.
- **Full Bayes:** the posterior $p(\theta\mid D) \propto p_\theta(D)\,p(\theta)$ as an object, not a point. Predictions marginalize: $p(y\mid x, D) = \int p_\theta(y\mid x)\,p(\theta\mid D)\,d\theta$. Intractable for deep nets; approximated by deep ensembles, SWAG, or Laplace — the practical source of epistemic uncertainty.

**Sufficiency:** a statistic $T(D)$ is sufficient if $p_\theta(D\mid T)$ is $\theta$-free (Fisher–Neyman: $p_\theta(D)=g_\theta(T(D))h(D)$). It is the formal sense in which a summary loses no information about $\theta$ — and the lens for asking what a learned representation must retain.

**Bias–variance of estimators** (distinct from the prediction decomposition revisited in *Learning theory*): $\operatorname{MSE}(\hat\theta)=\operatorname{Bias}^2+\operatorname{Var}$. MLE is asymptotically unbiased but can be high-variance in low data; MAP/shrinkage trades a little bias for large variance reduction — the statistical content of regularization.

### Frequentist vs Bayesian — two definitions of probability

| | Frequentist | Bayesian |
|---|---|---|
| Probability is | long-run frequency | degree of belief |
| $\theta$ is | fixed, unknown | random, has a posterior |
| Output | estimator + sampling distribution | posterior distribution |
| Interval | confidence (random interval) | credible (random parameter) |

They are not rivals so much as different conditioning. Frontier practice is hybrid: MLE pretraining (frequentist point estimate), Bayesian-flavored uncertainty heads, and frequentist eval protocols.

### Uncertainty quantification: confidence vs credible

The distinction is constantly conflated:

- **Confidence interval (CI):** a random interval with $P(\theta \in \text{CI}) \ge 1-\alpha$ over *repeated sampling*. For a fixed observed interval, $\theta$ is in it or not — the 95% is a property of the procedure.
- **Credible interval:** $P(\theta \in [\,a,b\,]\mid D) = 1-\alpha$ — a direct probability statement about $\theta$, conditional on the data and prior.

They coincide asymptotically (Bernstein–von Mises) but can differ sharply with informative priors or small $n$.

### Hypothesis testing & statistical power

A test fixes a null $H_0$, computes a statistic, and rejects when a $p$-value falls below $\alpha$ — controlling the **Type I** (false-positive) rate at $\alpha$. The neglected half is **power** $= 1-\beta = P(\text{reject}\mid H_1)$, the Type II control.

- Frontier relevance: comparing two models on a benchmark is a paired test. With $n$ eval items and effect size $d$, required $n$ grows as $\sim 1/d^2$ — tiny benchmarks are *underpowered*, so "no significant difference" usually means "couldn't detect one."
- Multiplicity: scanning many checkpoints/prompts/seeds inflates false positives; control FWER (Bonferroni) or FDR (Benjamini–Hochberg). The replication crisis is a multiplicity-and-power crisis, and ML leaderboards inherit it.
- The $p$-value is *not* $P(H_0\mid D)$ and *not* an effect size; report effect sizes with intervals.

### The bootstrap

Resample the data with replacement $B$ times, recompute the statistic, use the empirical distribution of $\hat\theta^{*(b)}$ as a stand-in for its sampling distribution. Assumption: the sample approximates the population (i.i.d.-ish); no parametric form needed. It is the default tool for putting CIs on benchmark accuracy, win-rates, and Elo when no closed form exists.

### The modern distribution-free toolkit

These are the methods built for frontier AI's regime — black-box models, abundant unlabeled data, continuous monitoring — emphasizing *what is guaranteed and under what assumption*.

- **Conformal prediction.** Wraps any predictor to emit a set $\hat C(x)$ with finite-sample marginal coverage $P(y_{n+1}\in\hat C(x_{n+1}))\ge 1-\alpha$. *Assumes only exchangeability* of calibration + test points — nothing about the model or data law. Split conformal: take quantile $q$ of calibration nonconformity scores $s(x,y)$, set $\hat C(x)=\{y: s(x,y)\le q\}$. Fails under distribution shift (fix: weighted conformal) and gives calibrated *sets*, not per-label probabilities. Deployed for LLM output sets and selective prediction.
- **Prediction-powered inference (PPI / PPI++).** When you have a small gold-labeled set plus many model-labeled points, PPI yields valid CIs for a population estimand (a mean, a coefficient) that are *tighter than gold-only* while remaining valid even if the model is biased — it uses labels to debias a "rectifier" term. *Assumes* the gold and unlabeled draws are i.i.d. from the same distribution. The right tool when human labels are the bottleneck and an LLM judge is cheap but imperfect.
- **Anytime-valid inference & e-values.** Classical $p$-values are only valid at a pre-specified $n$; peeking inflates error. An **e-value** $E\ge 0$ with $\mathbb{E}_{H_0}[E]\le 1$ supports a test that is valid *at every time simultaneously* (via Ville's inequality, $P(\sup_t E_t \ge 1/\alpha)\le\alpha$), and e-values multiply across independent experiments. Confidence sequences are their interval analog: valid under continuous monitoring and optional stopping. This is the statistically honest substitute for the ubiquitous "watch the eval curve and stop when it looks good" — essential for A/B tests and online RLHF/eval pipelines.

**One-line contrast:** bootstrap and CIs assume i.i.d. and target asymptotic validity; conformal needs only exchangeability and is finite-sample; PPI buys power from unlabeled data under i.i.d.; e-values trade a little power for validity under arbitrary stopping.

## Probabilistic modeling — distributions over structured worlds

Inference assumed a model was already given; this section is where you *write the model down*. Every frontier system is, formally, a parameterized distribution `p_θ` over a structured object — token sequences, pixels, molecular graphs, action trajectories — and *training is fitting that distribution, generation is sampling from it, reasoning is conditioning it*. What follows is the grammar for writing those distributions and the inference machinery for inverting them; the rest of frontier ML is special cases.

### Generative vs. discriminative — what you choose to model

- **Discriminative** models the conditional `p(y|x)` directly (or just a decision boundary). Cheaper, lower-variance when you only need the label. Classifiers, reward models, value heads.
- **Generative** models the joint `p(x, y)` (or `p(x)`), letting you sample, impute, score likelihood, and condition arbitrarily. Strictly harder, strictly more useful.
- The frontier insight: a generative model of `p(x)` over text *is* a universal conditional engine — `p(answer | question)` is just conditioning. An LLM is one giant generative model that subsumes discriminative tasks via prompting. The generative/discriminative line dissolved once `p(x)` got good enough to condition on anything.

### Latent-variable models and the ELBO

Introduce unobserved `z` to explain structure in `x`: `p_θ(x) = ∫ p_θ(x|z) p(z) dz`. The integral is intractable, so we bound the log-likelihood with a variational posterior `q_φ(z|x)`:

```
log p_θ(x) = ELBO(θ,φ;x) + KL(q_φ(z|x) ‖ p_θ(z|x))
ELBO = E_{q_φ}[log p_θ(x|z)]  −  KL(q_φ(z|x) ‖ p(z))
       └── reconstruction ──┘   └──── rate / regularizer ────┘
```

Because `KL ≥ 0`, the ELBO is a *lower bound* (Jensen, again); maximizing it tightens the posterior approximation and fits the model jointly. **EM** is the classical case: E-step sets `q = p_θ(z|x)` exactly (zeroing the KL), M-step maximizes the resulting expected complete-data log-likelihood. EM is coordinate ascent on the same ELBO — gradient methods just do both steps at once.

### Variational and amortized inference — the engine

- **Variational inference** turns inference into optimization: pick a family `q_φ`, maximize the ELBO. Mean-field factorizes `q`; expressive `q` (flows, structured posteriors) closes the bound.
- **Amortized inference** is the load-bearing modern move: instead of optimizing a separate `q` per datapoint, train *one* network `q_φ(z|x)` (an encoder) that emits posterior parameters in a forward pass. The **reparameterization trick** (`z = μ_φ(x) + σ_φ(x)⊙ε`, `ε∼N(0,I)`) makes the ELBO low-variance and differentiable.
- This is the conceptual bridge to in-context learning: a pretrained transformer doing ICL behaves like an *amortized Bayesian predictor* — it has learned to map a context (dataset) to a posterior predictive in one pass, exactly the amortization idea at sequence scale. (Developed in full under *Probabilistic reasoning*.)

### Graphical models — the conditional-independence backbone

A model is a factorization; the graph *is* the set of independence assumptions.

| Type | Factorization | Independence semantics | Frontier instance |
|---|---|---|---|
| Directed (Bayes net) | `∏ p(x_i \| pa(x_i))` | d-separation; causal-flavored | Autoregressive LMs (a chain), diffusion (a Markov chain), latent-variable decoders |
| Undirected (MRF/EBM) | `(1/Z) ∏ ψ_c(x_c)` | separation in the graph | Energy-based models, CRFs, attention as a soft fully-connected factor |

The cost of structure lives in the partition function `Z = ∫ ∏ψ` — tractable factorization (autoregressive, normalizing flows) vs. intractable `Z` (EBMs) is the dividing line that explains every architecture's training recipe.

### The generative zoo as one design space

Each family is a different answer to *"how do you make `∫ p(x|z)p(z)dz` or `Z` tractable?"* — trading off exact likelihood, sample quality, and sampling speed.

| Family | What it learns | Likelihood | Trade-off |
|---|---|---|---|
| **Autoregressive** | `∏_t p(x_t \| x_{<t})` via chain rule | exact, tractable | slow sequential sampling; the LLM/PixelCNN/audio default |
| **VAE** | ELBO bound on `log p(x)` w/ amortized `q_φ` | lower bound | blurry samples; great for representation/latents |
| **Normalizing flow** | exact `log p(x)=log p(z)+log\|det J\|` via invertible `f` | exact | architecture constrained to invertible w/ cheap Jacobian |
| **Energy-based (EBM)** | unnormalized `−E_θ(x)`, `Z` intractable | none directly | flexible; needs MCMC / score matching to train |
| **Diffusion / score** | the **score** `∇_x log p_t(x)` at noise levels `t` | bound (or exact via probability-flow ODE) | many sampling steps; current SOTA for images/video/molecules |
| **Flow matching** | a **velocity field** `v_t(x)` transporting noise→data | exact via ODE | fewer steps, simpler objective; the 2024–26 default |

### Diffusion, score, and flow matching — the unifying view

These three are *the same object* seen from different angles, and they dominate continuous-data frontier generation.

- **Score-based / diffusion**: corrupt data with a forward SDE `dx = f dt + g dW`, learn the score `s_θ(x,t) ≈ ∇_x log p_t(x)` by **denoising score matching** (predict the added noise; the loss is a tractable surrogate for the intractable score). Generate by running the reverse SDE, which needs only the score. This sidesteps `Z` entirely — you never normalize, you learn a gradient field.
- **Probability-flow ODE**: every diffusion SDE has a deterministic ODE with the *same marginals* `p_t`. Following it gives exact likelihoods (continuous change-of-variables) and connects diffusion to flows.
- **Flow matching / rectified flow**: directly regress a velocity field `v_θ(x,t)` whose ODE transports a simple base to data, via the **conditional flow matching** loss `E‖v_θ(x_t,t) − u_t(x_t|x_1)‖²` with `x_t=(1−t)x_0+t x_1`. Linear ("rectified") paths give straight trajectories → far fewer integration steps. Score-matching is the special case where the path is the Gaussian diffusion path; flow matching generalizes the target transport.
- **Why it matters in mid-2026**: flow matching is now the workhorse objective behind frontier image/video/audio generators and is moving into molecular and protein design (transporting over Riemannian/discrete manifolds), while **discrete diffusion** is a live challenger to autoregressive decoding for language. The common thread: *learn a vector field (score or velocity), integrate it, never touch the normalizer.*

### The throughline

Pick what to model (`p(x)` vs `p(y|x)`), declare the factorization (the graph), choose how to dodge intractability (exact likelihood, an ELBO, or a learned vector field), and inference — amortized into a single network — does the rest. Pretraining maximizes likelihood; diffusion/flow sampling and ICL are inference under this same calculus.

## Learning theory — generalization as a statistical question

We can now fit distributions and run inference — but does minimizing loss on a finite sample actually lower loss on the population? That is the generalization question: whether the empirical estimator concentrates. Frontier AI is built on the empirical fact that absurdly overparameterized models generalize anyway, which breaks classical theory and forces a statistical reframing of *why*.

### Learning as inference
Supervised learning is estimation of a conditional `p(y|x)`. The training objective is a plug-in for a population risk:
- **Risk:** `R(f) = E_{(x,y)~D}[ℓ(f(x),y)]` — what we want low.
- **Empirical risk:** `R̂(f) = (1/n) Σ ℓ(f(xᵢ),yᵢ)` — what we can measure.

ERM picks `f̂ = argmin_{f∈F} R̂(f)`. With log-loss this *is* maximum likelihood, so pretraining is ERM over next-token cross-entropy. The generalization gap `R(f̂) − R̂(f̂)` is the statistical object the whole field is implicitly bounding.

### Bias–variance and its breakdown
Classical decomposition of *prediction* error: `error = bias² + variance + noise`. Capacity ↑ ⇒ bias ↓, variance ↑ ⇒ U-shaped test error, "don't interpolate noise." Frontier models violate this.
- **Double descent** (Belkin 2019; Nakkiran 2020): test error rises to a peak at the *interpolation threshold* (params ≈ samples), then **descends again** in the overparameterized regime. Holds across model size, data size, and training time ("epoch-wise" double descent).
- **Benign overfitting** (Bartlett 2020): a model can fit pure noise to zero training error yet generalize — the excess capacity absorbs noise in directions orthogonal to signal. Requires a specific effective-rank spectrum of the data covariance; not automatic.
- Takeaway: capacity is not the right complexity axis. The relevant quantity is which interpolating solution the optimizer selects.

### PAC and PAC-Bayes (KL as complexity)
PAC bounds the gap by hypothesis-class complexity. Schematically, w.p. ≥ 1−δ:
`R(f̂) ≤ R̂(f̂) + O(√((complexity(F) + log(1/δ)) / n))`
with `complexity` = VC dimension, Rademacher complexity, or covering number. **These bounds are vacuous for deep nets** (complexity ≫ n) and are uniform over `F`, which is the wrong granularity.
- **PAC-Bayes** fixes the granularity. For posterior `Q` over weights and prior `P`:
`E_{w~Q}[R(w)] ≤ E_{w~Q}[R̂(w)] + √((KL(Q‖P) + log(n/δ)) / (2n))`
Complexity becomes `KL(Q‖P)` — how far the trained distribution moved from initialization. This yields the only **non-vacuous** generalization bounds for real (small) nets (Dziugaite & Roy 2017), and connects directly to flat-minima and compression arguments: solutions you can describe cheaply relative to the prior generalize.

### NTK vs. feature-learning regimes
Two limits of wide-network training:
| Regime | Behavior | Generalization story |
|---|---|---|
| **NTK / lazy** (Jacot 2018) | Weights barely move; net ≈ linearization at init; training = kernel regression with the (fixed) neural tangent kernel | Convex, fully analyzable; explains convergence — but features are *fixed*, so it cannot explain representation learning |
| **Feature-learning / μP** (Yang & Hu, Tensor Programs) | Weights move O(1); kernel evolves; representations adapt to data | The regime frontier models actually live in; μP gives the correct init/LR scaling so the feature-learning limit is stable and hyperparameters transfer across scale |

NTK is the right model for *infinitely wide, lazily trained* nets and a clean baseline; it is the wrong model for why transformers learn reusable circuits. Real training is somewhere in between but firmly outside the lazy regime at scale.

### Implicit bias of SGD
With many interpolating solutions, the optimizer is a *prior*. For separable data, gradient descent on logistic/exponential loss converges to the **max-margin (hard-margin SVM) solution** (Soudry 2018) — implicit ℓ₂ regularization with no explicit penalty. More broadly: SGD noise, finite step size, and the **edge of stability** (sharpness hovers near `2/η`) bias toward flat minima, which PAC-Bayes ties back to generalization. This is why architecture + optimizer matter more than the regularizer: they select *which* zero-training-loss function you get.

### Scaling laws as empirical statistics
Scaling laws are not a theorem; they are a **measured regularity** — loss as a power law, fit by regression over many runs:
`L(N,D) ≈ E + A·N^{-α} + B·D^{-β}` (Chinchilla form; `E` = irreducible entropy of text).
- **Kaplan 2020 → Chinchilla (Hoffmann 2022):** for fixed compute `C ≈ 6ND`, optimal `N* ∝ C^{0.5}`, `D* ∝ C^{0.5}` — scale params and data ~equally. Kaplan under-counted data.
- The exponents are **estimates**, not constants — they shift with data quality, tokenizer, architecture, and domain (code/math/proteins re-fit). Treating them as universal is the standard error.
- **Inference-optimal (2024+):** Chinchilla minimizes *training* compute; if you serve the model billions of times, over-train a smaller model. 2024–25 frontier models (Llama 3, Phi, etc.) deliberately over-train past Chinchilla.

### Emergence: real or metric artifact?
"Emergent abilities" — capabilities that appear abruptly at scale (Wei 2022) — were challenged by Schaeffer 2023: discontinuities largely vanish under **smooth, per-token metrics** (e.g., log-likelihood) and reappear only under harsh, thresholded metrics (exact-match accuracy). Mid-2026 synthesis: most "emergence" is a **measurement nonlinearity** — the underlying loss improves smoothly and predictably; the *task metric* is a step function of that loss. The open residue is genuinely discontinuous in-context phenomena (e.g., induction-head formation, grokking-style phase transitions) that are sharp in the loss curve itself, not just the metric.

### What classical theory explains — and where it breaks
- **Explains:** consistency of ERM/MLE, convergence under convexity (NTK), the underparameterized side of the U-curve, margin-based generalization.
- **Breaks:** uniform-convergence/VC bounds (vacuous), the bias–variance U-curve (double descent), "interpolation = overfitting" (benign overfitting), and any account that ignores the optimizer.
- **The working modern picture:** generalization is decided by the *implicit bias of (architecture, optimizer, data spectrum)* selecting a low-complexity (flat / low-`KL`-from-prior) interpolant — best bounded by PAC-Bayes and best *predicted*, in practice, by scaling laws.

## Probabilistic reasoning & decision-making

If pretraining is maximum likelihood and generalization is the guarantee that it transfers, then *acting* on a model — updating beliefs, choosing the next query, deciding what to ship — is decision theory under uncertainty. This section is the spine that turns a fitted distribution into an agent: it is where "what does the model believe?" becomes "what should it do, and what would happen if it intervened?"

### Bayesian reasoning & sequential updating

The whole edifice is one identity, applied recursively:

```
p(θ | D) ∝ p(D | θ) p(θ)        posterior ∝ likelihood × prior
p(θ | D₁, D₂) ∝ p(D₂ | θ) p(θ | D₁)   yesterday's posterior is today's prior
```

- **Conjugacy / closed form** (Beta–Bernoulli, Normal–Normal) is the exception; at scale we *approximate* the posterior — variational inference (ELBO maximization), MCMC/HMC, Laplace, or **amortized** inference where a network outputs the posterior in one forward pass.
- **Frontier instance:** PFNs / TabPFN reframe Bayesian inference as "pretrain on a task prior, predict the posterior predictive in a single forward pass" — the amortized limit of hierarchical Bayes. **HiBayES** does sequential/partial pooling over eval items rather than collapsing to a flat mean.
- **Credible ≠ confidence** (see *Statistical inference*). A credible interval is a posterior-probability statement about θ; a confidence interval is a coverage statement about the procedure. Know which you are bolding.

### Decision theory & expected utility

A belief is inert until paired with a loss. The decision rule is the Bayes action:

```
a* = argmin_a  E_{θ ~ p(θ|D)} [ L(θ, a) ]
```

| Loss L(θ,a) | Optimal Bayes action |
|---|---|
| Squared error | posterior **mean** |
| Absolute error | posterior **median** |
| 0–1 | posterior **mode** (MAP) |

This is why a calibrated posterior matters operationally: an abstention head, a router choosing a model tier, or a safety filter is choosing the action minimizing expected loss under a cost asymmetry (a false "safe" costs more than a false refusal). Proper scoring rules (log-loss, Brier) are exactly the losses honest probabilities minimize — they cannot be gamed by misreporting confidence.

### Exploration vs. exploitation

When the action *changes what you observe*, greedy is provably suboptimal — you must pay information cost now for reward later.

- **Bandits:** minimize regret with **UCB** (optimism: `argmax_a μ̂_a + √(2 ln t / n_a)`) or **Thompson sampling** (act greedily w.r.t. a posterior sample — Bayesian exploration for free). **Frontier instance:** LMArena's *active sampling* of informative matchups is a dueling bandit over models.
- **RL-as-inference / control-as-inference:** introduce an optimality variable `O_t` with `p(O_t=1 | s,a) ∝ exp(r(s,a))`; conditioning the trajectory on optimality and doing variational inference *recovers* maximum-entropy RL. The KL-regularized objective at the heart of post-training,

```
max_π  E_π[r(x,y)] − β · KL(π ‖ π_ref),
```

has a closed-form solution `π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)` — i.e. RLHF/DPO is **Bayesian posterior reweighting of the base policy**, with the reference model as prior and reward as log-likelihood. Entropy bonuses, the GRPO/PPO KL term, and reasoning-RL all sit inside this frame.

### Active learning & optimal experimental design

Same machinery, pointed at *which data to acquire*: choose the query that most reduces posterior uncertainty.

- **Information-theoretic objective:** maximize expected information gain `I(θ; y | x) = H(θ) − E_y[H(θ|y)]` (BALD), or classical D-/A-optimality (max log-det of the Fisher information). The active matchup selection above is this for evals.
- **Frontier instances:** RLHF *preference-pair selection*, active distillation/data curation, sequential prompt optimization, and self-improvement loops where the model proposes the next training example are all optimal-design problems — the bottleneck has moved from model fitting to *which datum is worth a label*.

### Causal inference — the rung above association

Prediction lives on correlation; agents that *intervene* need a higher rung. Pearl's ladder:

| Rung | Question | Object |
|---|---|---|
| Association | `P(Y \| X)` | seeing |
| Intervention | `P(Y \| do(X))` | doing |
| Counterfactual | `P(Y_x \| X=x', Y=y')` | imagining |

- **Two languages, one content.** Potential outcomes (Neyman–Rubin): each unit has `Y(1), Y(0)`, only one observed — causal inference is a missing-data problem (ATE `= E[Y(1)−Y(0)]`). Structural/graphical (Pearl): a DAG plus **do-calculus**, whose three rules decide when `P(Y|do(X))` is *identifiable* from observational data (back-door / front-door criteria).
- **Identification before estimation.** Ignorability `{Y(0),Y(1)} ⟂ T | X` + overlap `0 < P(T=1|X) < 1` license `ATE = E_X[E[Y|T=1,X] − E[Y|T=0,X]]`. Then estimate: **DML** (Neyman-orthogonal scores + cross-fitting) gives √n-valid CIs with arbitrary ML nuisances; causal forests give CATEs.
- **The collider trap:** conditioning on more variables is *not* safer — adjusting for a collider or mediator *creates* bias. Draw the DAG first.
- **Frontier instance:** shipping a model change *is* an intervention; "the metric went up" is a confounded observational claim unless the rollout was randomized — acute for agentic products pushing continuous online changes.

### In-context learning as implicit Bayesian inference

The unification that makes the thesis bite, and the payoff of the amortized-inference thread from *Probabilistic modeling*. Pretraining on a mixture of latent tasks/documents makes the model approximate the **posterior predictive** over a latent concept `z`:

```
p(y | x, prompt) = ∫ p(y | x, z) · p(z | prompt) dz
```

The demonstrations in the context don't update weights — they sharpen the posterior `p(z | prompt)`, so few-shot prompting *is* sequential Bayesian updating run in the forward pass (Xie et al.'s implicit-Bayes view; mechanistically realized as induction heads and, in linear-attention regimes, as in-context gradient descent / amortized regression). Chain-of-thought and inference-time compute extend this: spending more tokens is **test-time marginalization** over reasoning paths, and self-consistency is Monte-Carlo estimation of the answer posterior `argmax_y Σ_paths p(y, path | x)`.

**The through-line:** reasoning is inference under uncertainty. Updating beliefs (Bayes), choosing actions (expected utility), choosing what to observe (exploration / experimental design), and reasoning about interventions (causality) are one calculus — and every frontier loop, from RLHF to o-series test-time search, is a special case of it.

## Where it all shows up at the frontier

Everything above is not analogy: frontier AI *is* probabilistic inference executed at scale. The same objects — likelihoods, posteriors, KL divergences, proper scoring rules, confidence sets — appear under engineering names (pretraining, RLHF, diffusion, evals, calibration). Naming the statistics behind the system is what lets you reason about when it works, when it fails, and what its outputs actually guarantee.

### Pretraining is maximum likelihood

Next-token pretraining minimizes cross-entropy, which is exactly the MLE of an autoregressive model `p_θ(x) = ∏_t p_θ(x_t | x_{<t})`:

```
L(θ) = −E_{x∼data} [ Σ_t log p_θ(x_t | x_{<t}) ]
```

Minimizing this is minimizing `KL(data ‖ p_θ)` up to the (fixed) data entropy. Cross-entropy is a *strictly proper scoring rule*, so the minimizer is the true conditional distribution — the model is calibrated **in-distribution by construction**, and miscalibration is a generalization/shift phenomenon, not an objective defect. Scaling laws are the empirical statement that this MLE loss falls as a power law in `(N, D, C)`; perplexity is just `exp(L)`.

### Alignment is KL-regularized inference and preference modeling

Post-training does *not* re-estimate a likelihood; it does constrained inference against the pretrained prior `π_ref`.

- **RLHF / PPO** optimizes reward subject to a KL leash: `max_π E_π[r(x,y)] − β·KL(π ‖ π_ref)`, whose closed-form optimum is the tilted/Gibbs posterior `π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)`. The KL term is a Bayesian prior, not a hack.
- **Reward models** are **Bradley–Terry** logistic models of pairwise preference: `P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))`.
- **DPO** (Rafailov et al., 2023) substitutes the closed-form `π*` back into the BT likelihood, turning RLHF into a *single* classification loss on preference pairs — no sampling, no separate reward model. It is preference MLE with the policy as implicit reward.
- **RLVR / GRPO** (the o-series, DeepSeek-R1, and the 2025–26 reasoning wave) swaps the learned reward for a *verifiable* one (unit tests, proof checkers, exact-match), making the reward unbiased and the optimization a low-variance policy-gradient estimate. GRPO replaces the value baseline with a group-relative advantage — a variance-reduction trick, i.e. a control variate.

### Generative modeling is score / probability-path estimation

- **Diffusion** trains by **denoising score matching**: learn `s_θ(x,t) ≈ ∇_x log p_t(x)`, the score of the noised marginal, via `E ‖ s_θ(x_t,t) − ∇ log p(x_t|x_0) ‖²`. Sampling integrates the reverse SDE/probability-flow ODE. Generation = following an estimated score field.
- **Flow matching** (Lipman et al., 2023; the backbone of SD3, Flux, and most 2024–26 image/video/audio models) regresses a velocity field onto a chosen **probability path** `p_t` between noise and data: `E ‖ v_θ(x_t,t) − u_t(x_t|x_1) ‖²`. Straight (rectified) paths give few-step sampling. Both are MLE-adjacent: continuous flows have a tractable change-of-variables likelihood; the regression targets are conditional and unbiased.

### In-context learning is amortized Bayes

A frozen LLM doing few-shot ICL behaves like an approximate Bayesian predictor: conditioning on a prompt `D` and querying `x` approximates the posterior predictive `p(y | x, D) = ∫ p(y|x,θ) p(θ|D) dθ`, with the integral *amortized* into a single forward pass by the pretraining meta-distribution. This is explicit in **PFNs / TabPFN** (Hollmann et al.; TabPFN v2, *Nature* 2025), which are trained on millions of synthetic datasets to emit a calibrated posterior predictive for a whole tabular dataset in one pass — Bayesian inference with the integral baked into the weights. The cost of inference moves to training; the payoff is constant-time "fitting."

| Foundation | Frontier instance (2024–2026) | Statistical object |
|---|---|---|
| Maximum likelihood | Next-token pretraining, cross-entropy | `argmax_θ Σ log p_θ` |
| Proper scoring rules | Perplexity, log-loss evals | Cross-entropy / Brier |
| KL-constrained optimization | RLHF/PPO, DPO | Gibbs posterior `π_ref e^{r/β}` |
| Logistic preference model | Reward models | Bradley–Terry |
| Verifiable-reward policy grad | RLVR, GRPO (o-series, R1) | Control-variate REINFORCE |
| Score matching | Diffusion (sampling SDE/ODE) | `∇ log p_t` |
| Probability paths | Flow matching (SD3, Flux) | Continuous normalizing flow |
| Posterior predictive | ICL, PFNs / TabPFN v2 | Amortized `p(y\|x,D)` |
| Hypothesis testing | Benchmark/Arena evals | Paired tests, bootstrap CIs |
| Uncertainty quantification | Calibration, hallucination, abstention | Selective prediction, ECE |
| Distribution-free coverage | Conformal LM, PPI | Exchangeable prediction sets |

### Evaluation is a statistical experiment

A benchmark score is an estimate with a sampling distribution, not a number. Treating it as exact is the dominant methodological error in 2024–26 leaderboard culture.

- Report **confidence intervals** (Wilson/bootstrap) on accuracy; a 0.4-point gap on 1k items is noise.
- Compare models with **paired** tests (McNemar, paired bootstrap) on shared items, not marginal accuracies — this kills most variance.
- LMSYS/Chatbot-Arena Elo is a **Bradley–Terry fit to pairwise human votes**; rankings carry CIs and shift under sampling.
- LLM-as-judge introduces a *measurement model* with bias and variance; debias and bound it rather than trusting raw win rates.

### Uncertainty quantification is what makes deployment trustworthy

The frontier-safety vocabulary maps cleanly onto UQ.

- **Calibration**: does a stated 0.8 mean 80% correct? Measured by ECE / reliability diagrams; RLHF is known to *degrade* the calibration that pretraining produces.
- **Hallucination & selective prediction**: abstention is the optimal action under a cost on errors — answer only when `confidence ≥ τ`, trading coverage for risk along a risk–coverage curve.
- **Conformal prediction** gives finite-sample, distribution-free coverage `P(y ∈ Ĉ(x)) ≥ 1−α` under exchangeability only — used for LLM output sets with guarantees and for bounding agent/tool errors.
- **Prediction-Powered Inference** (Angelopoulos et al., 2023) lets you do valid statistical inference (CIs, means) using cheap model labels plus a small gold set — increasingly the principled way to evaluate and to use models as measurement instruments without inheriting their bias.

The throughline: pretraining estimates a distribution, alignment performs constrained inference against it, generation integrates an estimated field, evaluation is experiment design, and UQ is the contract you ship. Frontier AI is applied probability — the only question is whether you account for the uncertainty explicitly or let it surprise you in production.

## How to study this

The sections are ordered as a dependency chain, and that *is* the recommended sequence — each rung assumes the one below it.

1. **Probability first, until it is reflex.** You cannot read the rest without fluency in expectation/variance (gradient estimation), Bayes (everything inferential), KL and cross-entropy (the loss and the regularizer), and concentration (why evals mean anything). Bishop's *Pattern Recognition and Machine Learning* (Ch. 1–2) or MacKay's *Information Theory, Inference, and Learning Algorithms* (free, and the information-theory framing is exactly this document's spine) are the two highest-leverage texts.
2. **Statistical inference, with an eye on the distribution-free toolkit.** Estimation and the confidence/credible distinction are classical; conformal prediction, PPI, and anytime-valid inference are the genuinely modern, frontier-relevant additions — Angelopoulos & Bates' *gentle introduction to conformal prediction* is the single best entry point, and the PPI papers are short.
3. **Probabilistic modeling and learning theory together.** The ELBO and the generative zoo (modeling) and double descent / PAC-Bayes / scaling laws (theory) reinforce each other; Murphy's *Probabilistic Machine Learning* (two volumes) covers both at the right altitude. For the diffusion/score/flow unification, read the score-SDE paper (Song et al.) then flow matching (Lipman et al.) back-to-back.
4. **Reasoning and decision-making.** Sutton & Barto for the RL/bandit core; Pearl's *Causality* (or the shorter *Book of Why* for intuition) for the causal ladder. The RL-as-inference and ICL-as-Bayes connections are best absorbed from the primary papers once the machinery above is in place.
5. **The frontier mapping last** — it is the payoff, and it only lands once the foundations are reflexive. Re-read *Where it all shows up at the frontier* and check that every engineering term resolves to a statistical object without effort. If it does, you have the spine.

Highest-leverage shortcut: MacKay + Murphy (vol. 1) + the conformal tutorial + the score-SDE and flow-matching papers will carry you most of the way. Everything else is depth on a rung you already understand.

## See also

[Data Foundations for Frontier AI](data-foundations.md) · [Applied Statistics for the Modern AI Era](applied-statistics-ai-era.md) · [APSL — Algorithmic Probabilistic Structure Learning](apsl-structure-learning.md) · [Frontiers in Machine Learning](frontiers-ml.md) · [Curriculum Index](curriculum-index.md) · [Concept library (bricks)](../library/bricks/README.md) · [Library: Foundations](../library/foundations/README.md)
