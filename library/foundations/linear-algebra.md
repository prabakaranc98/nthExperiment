# Linear Algebra for ML

*The version you need to read frontier papers — not a general course.*

---

## Why this matters at the frontier

- **LoRA** works because weight updates are low-rank: ΔW = AB where A ∈ ℝ^{d×r}, B ∈ ℝ^{r×k}, r << min(d,k). The rank constraint is the entire idea.
- **Neural Tangent Kernel (NTK)** theory is fundamentally about the eigenspectrum of the kernel matrix — which eigenvalues are large determines what the network learns first.
- **Attention** computes QKᵀ — a bilinear form. Multi-head attention applies multiple low-rank projections. Understanding why GQA and MLA reduce KV cache requires understanding low-rank factorization.
- **PCA, SVD, and matrix decompositions** appear constantly in interpretability (feature geometry, principal components of activations) and in theoretical analyses.
- **Matrix calculus** is unavoidable — gradients of loss w.r.t. weight matrices, Jacobians, the Fisher information matrix.

---

## Vectors and inner products

A vector x ∈ ℝ^n is a point in n-dimensional space. The **inner product (dot product)** x·y = Σᵢ xᵢyᵢ measures alignment:
- x·y = ‖x‖ ‖y‖ cos(θ) where θ is the angle between them
- x·y = 0 → orthogonal (no alignment)
- x·y = ‖x‖² → y is exactly x (maximum alignment per unit length)

**Why it matters:** attention computes QKᵀ — row i of Q dotted with row j of K gives the attention score between token i and j. High score = high alignment = attend strongly.

A vector can also represent a **direction** in a feature space. In the superposition hypothesis, features are directions in activation space — you measure how much a concept is present by taking the inner product of the activation with the feature direction.

---

## Matrices as linear transformations

A matrix W ∈ ℝ^{m×n} transforms vectors: y = Wx maps x ∈ ℝ^n to y ∈ ℝ^m.

The columns of W are where the standard basis vectors go. The row space and column space tell you what W can and cannot represent.

**Rank:** the number of linearly independent rows (= columns). A rank-r matrix has an r-dimensional image — it "collapses" the input onto an r-dimensional subspace.

**Rank deficiency in RL:** when a policy's gradient has low rank, it's only updating along a few directions in parameter space. This can be a sign of training collapse.

---

## Eigendecomposition

For a square symmetric matrix A, eigendecomposition is: A = QΛQᵀ

where Q = orthogonal matrix of eigenvectors, Λ = diagonal matrix of eigenvalues.

An eigenvector v satisfies Av = λv — it's a direction that A only *scales*, not rotates.

**Why it matters at the frontier:**

**NTK (Neural Tangent Kernel):** in the infinite-width limit, a neural network's training dynamics are governed by K = J·Jᵀ where J is the Jacobian (gradient matrix). The eigenvalues of K determine learning speed — features aligned with large eigenvectors are learned first, small eigenvectors last. This is the "spectral bias" and explains why neural nets learn low-frequency functions first.

**Loss landscape:** the Hessian H = ∇²L has eigenvalues that determine the curvature of the loss surface. Large eigenvalues = sharp curvature = sensitive directions. Edge of stability says gradient descent runs near the maximum eigenvalue of the Hessian.

**Sharpness-Aware Minimization (SAM)** directly minimizes the maximum eigenvalue of H (the sharpness) as a regularizer toward flat minima.

---

## Singular Value Decomposition (SVD)

For any matrix M ∈ ℝ^{m×n}: M = UΣVᵀ

- U ∈ ℝ^{m×m}: left singular vectors (orthonormal columns)
- Σ ∈ ℝ^{m×n}: diagonal, singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
- V ∈ ℝ^{n×n}: right singular vectors (orthonormal columns)

**Low-rank approximation:** keep only the top-r singular values/vectors:
M ≈ Uᵣ Σᵣ Vᵣᵀ (minimizes Frobenius norm among rank-r matrices)

**LoRA:** fine-tuning updates ΔW are constrained to be low-rank: ΔW = AB. This is motivated by the observation that weight update matrices during fine-tuning have low "intrinsic rank" — the important updates live in a low-dimensional subspace. SVD analysis of ΔW shows that the top few singular values capture most of the variance.

**MLA (Multi-head Latent Attention):** DeepSeek's KV cache compression uses low-rank projection: instead of storing K, V at full dimension, store a shared compressed latent c = xWc then reconstruct K = cWk, V = cWv. Cuts KV cache dramatically.

---

## Matrix calculus (the version you actually need)

Let L be a scalar loss, W a weight matrix. What is ∂L/∂W?

**Key identities:**
- If L = aᵀWb, then ∂L/∂W = abᵀ (outer product)
- If L = tr(AW), then ∂L/∂W = Aᵀ
- Chain rule: if y = Wx and L = f(y), then ∂L/∂W = (∂L/∂y) · xᵀ

The gradient ∂L/∂W has the same shape as W. The entry (∂L/∂W)ᵢⱼ is how much L changes if Wᵢⱼ increases by a small amount.

**Jacobian:** if f: ℝ^n → ℝ^m, the Jacobian J ∈ ℝ^{m×n} has Jᵢⱼ = ∂fᵢ/∂xⱼ. This generalizes the gradient to vector-valued functions and is central to understanding how transformations compose in deep networks.

**Fisher Information Matrix:** F = E[∇log p(x;θ) · ∇log p(x;θ)ᵀ] — the covariance of the score function. F captures the curvature of the KL divergence at the current parameters. Natural gradient descent uses F⁻¹ instead of the identity as a preconditioner — it's the second-order optimizer that respects the geometry of the probability simplex. TRPO in RL uses a trust region defined in terms of F.

---

## Norms and distances

- **L2 norm:** ‖x‖₂ = √(Σ xᵢ²) — Euclidean distance
- **L1 norm:** ‖x‖₁ = Σ |xᵢ| — promotes sparsity
- **Frobenius norm:** ‖A‖_F = √(Σᵢⱼ Aᵢⱼ²) = √(tr(AᵀA)) — "flat" matrix norm
- **Spectral norm:** ‖A‖₂ = σ_max(A) — largest singular value; relevant for Lipschitz constraints and GAN stability

**Weight decay / L2 regularization:** adds λ‖W‖_F² to the loss. AdamW decouples this from the adaptive step, making the effective regularization cleaner.

---

## The geometry of representations

Representations in modern LLMs are high-dimensional vectors. The geometry matters:

**Cosine similarity:** cos(u, v) = u·v / (‖u‖ ‖v‖) — angular similarity, ignoring magnitude. Used in retrieval, contrastive learning evaluation.

**Alignment & uniformity:** two properties of good contrastive representations. Alignment: positive pairs should be close (high cosine similarity). Uniformity: representations should be spread evenly on the hypersphere (high entropy). These trade off against each other.

**Isotropy vs. anisotropy:** language model representations are famously anisotropic — they don't spread evenly in all directions, and most variance is in a few dimensions. This degrades retrieval quality and is why representation calibration (whitening, etc.) sometimes helps.

**Linear representation hypothesis:** a strong empirical finding in mechanistic interpretability — concepts are often represented as linear directions in activation space. The "king - man + woman = queen" analogy, but at the level of actual model internals. This is what makes sparse autoencoders possible: if features are linear directions, you can find them by dictionary learning.

---

## Quick reference

| Concept | Frontier appearance |
|---------|-------------------|
| Inner product | Attention scores: Q·Kᵀ; feature activation in superposition |
| Rank / low-rank | LoRA (ΔW = AB); MLA (KV compression); NTK eigenspectrum |
| SVD | Low-rank approximation; analyzing ΔW structure |
| Eigendecomposition | NTK learning dynamics; Hessian curvature; SAM |
| Matrix calculus | Backpropagation; gradient derivations |
| Fisher information | Natural gradient; TRPO trust region; second-order optimization |
| Spectral norm | GAN training stability; Lipschitz constraints |
| Cosine similarity | Contrastive learning; retrieval; alignment metric |
