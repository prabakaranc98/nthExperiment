# Statistical Learning Theory

*Generalization for overparameterized models — the modern theory, not classical VC dimension.*

---

## Why classical theory breaks

Classical statistical learning theory (VC dimension, Rademacher complexity, uniform convergence) gives generalization bounds of the form:

**Test error ≤ Training error + C · √(complexity/n)**

where complexity is related to model capacity and n is the number of training examples.

**The problem:** modern neural networks have:
- More parameters than training examples (overparameterized)
- Zero training error (they interpolate — perfectly fit the training data)
- Yet they *generalize* to test data

Classical bounds predict catastrophic overfitting. We observe excellent generalization. The bounds are vacuous — they give test error bounds larger than 1 (meaningless for a probability).

This is the central puzzle that the modern theory of generalization is trying to resolve.

---

## The interpolation threshold and double descent

**Classical bias-variance tradeoff:** as model complexity grows, test error follows a U-curve — initially decreases (less bias), then increases (more variance / overfitting).

**Double descent (Belkin et al., 2019; Nakkiran et al., 2019):** the U-curve *continues past the classical minimum* with a second descent into the overparameterized regime.

```
Test error
    |          Classical minimum
    |         /\
    |        /  \
    |       /    \          Double descent
    |      /      \        /
    |_____/        \______/________
                  ↑
         Interpolation threshold
         (parameters = data points)
```

At the interpolation threshold, test error spikes (the model is struggling to both fit the data and generalize). Past it — in the overparameterized regime — test error falls again and can surpass the classical minimum.

**Why?** In the overparameterized regime, the model has many zero-training-loss solutions. Gradient descent (via its implicit bias) selects the minimum-norm one among them. This minimum-norm interpolating solution generalizes because it makes the "smoothest" possible fit to the data.

**Frontier relevance:** all modern large language models are massively overparameterized. Their generalization cannot be explained by classical theory and is exactly what modern theory is trying to characterize.

---

## Benign overfitting

A precise theoretical characterization (Bartlett et al., 2020 for linear regression; ongoing work for neural networks):

**Benign overfitting occurs when:**
1. The model perfectly fits training data (zero training error)
2. Yet achieves near-optimal test error

**The mechanism in linear models:** decompose the data into a "signal" direction and "noise" directions. If the noise directions are high-dimensional enough (the covariance matrix has many small eigenvalues), the interpolating solution can fit the noise with negligible perturbation to the signal prediction.

**Conditions required:**
- Enough directions of near-zero eigenvalue in the covariance (high-dimensional data)
- Minimum-norm interpolation (which SGD approximately achieves)

**The key quantity:** the ratio of effective rank to the number of data points. When this is large, benign overfitting holds.

**For neural networks:** the picture is more complex because the effective "feature space" changes during training. The NTK regime gives a linear approximation where benign overfitting theory applies. The feature-learning regime requires richer analysis.

---

## PAC learning and the modern framework

**PAC (Probably Approximately Correct) learning:** a hypothesis class H is PAC-learnable if there exists an algorithm that, for any ε, δ > 0, outputs a hypothesis h with test error < ε with probability ≥ 1-δ, using poly(1/ε, 1/δ) samples.

**Sample complexity:** the minimum n needed for ε, δ guarantee. For VC dimension d: n = O(d/ε + log(1/δ)/ε).

**Why VC dimension is insufficient for deep networks:** VC dimension of a neural network with W weights is O(W log W) — enormous. Bounds based on VC dimension predict test error ≈ 1 for any reasonable-sized model. Empirically, models generalize far better than this.

---

## PAC-Bayes bounds — the ones that actually hold

**PAC-Bayes** (McAllester, 1999): instead of bounding the risk of a single hypothesis, bound the risk of a *posterior distribution* Q over hypotheses relative to a prior P:

```
E_{h~Q}[L(h)] ≤ E_{h~Q}[L̂(h)] + √[(KL(Q||P) + log(2√n/δ)) / (2n)]
```

**Why this is better:**
- Q can be a distribution over networks, not a single network
- KL(Q||P) measures how much Q differs from the prior — a data-dependent complexity measure
- Dziugaite & Roy (2017) showed that by choosing Q as a Gaussian perturbation around a trained network, you get *non-vacuous* (< 1) bounds for the first time

**Frontier relevance:** PAC-Bayes bounds are the theoretical tool that connects "flat minima generalize better" (SAM) to a formal guarantee. SAM implicitly minimizes the KL term by seeking weights where nearby perturbations don't increase the loss.

---

## Implicit bias of gradient descent as regularization

The most practically important theoretical insight:

**Fact:** gradient flow on overparameterized linear regression converges to the minimum-norm interpolating solution — the one closest to initialization in L2 norm.

**Implication:** SGD is implicitly regularized, even without explicit regularization. The choice of optimizer, learning rate, batch size, and parameterization all affect *which* minimum is found, acting as an implicit prior.

**Neural network case (ongoing):**
- SGD prefers low-rank solutions (connects to LoRA being effective)
- Large learning rates prefer flatter minima
- Batch size acts as implicit learning rate scaling (linear scaling rule)
- Parameterization matters: μP vs. standard vs. NTK parameterization give different implicit biases at scale

**The practical implication:** when you tune a neural network, you're not just optimizing — you're also implicitly selecting from the space of interpolating solutions. Understanding the implicit bias tells you what kind of solutions your hyperparameter choices favor.

---

## Grokking — a case study in generalization dynamics

**Observation (Power et al., 2022):** train a small transformer on modular arithmetic (a + b mod p). Training loss drops quickly to zero. Test loss *remains high* for thousands of additional steps, then suddenly drops.

**The mechanism (identified by interpretability):**
- Early in training: the network memorizes training examples (generalizing circuit AND memorizing circuit both present, memorizing dominates)
- Weight decay slowly penalizes the memorizing circuit (large weights) more than the generalizing circuit (small, structured weights)
- Phase transition: when the generalizing circuit becomes sufficient, the model "clicks" into generalization

**Why it matters:**
1. Generalization can be *latent* — the model might "know" the right algorithm before it expresses it
2. Weight decay is doing real work even when training loss is already zero
3. Provides a clean model where interpretability methods can identify the phase transition mechanistically
4. Suggests that "grokking" in LLMs (sudden capability acquisition during training) might have a similar structure
