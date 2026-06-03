# Optimization: Deep Theory

*Not just "gradient descent goes downhill" — the version that explains frontier training behavior.*

---

## Why this matters at the frontier

- **Edge of stability** — frontier models are trained with learning rates larger than classical theory allows. Understanding why training still converges (and how to control it) requires the edge-of-stability framework.
- **Implicit bias** — SGD and Adam don't just find *a* minimum; they find specific ones. Which one matters enormously for generalization. This is the hidden regularizer in every training run.
- **μP (Maximal Update Parameterization)** — hyperparameters found at 7M parameters transfer perfectly to 70B, if you parameterize correctly. This requires understanding how the learning signal scales with width.
- **Loss spikes and instability** — FP8 training, trillion-scale training, and MoE routing all produce loss spikes. Understanding why — and how MuonClip fixes it — requires the eigenvalue picture.
- **Adam vs. SGD vs. Muon** — these aren't just "different optimizers," they're different implicit biases. Their geometry determines what solutions they prefer.

---

## The gradient descent trajectory

The classic picture: θ ← θ - η·∇L is too simple. Real neural network training has richer dynamics.

**Gradient flow (continuous time limit):**
```
dθ/dt = -∇L(θ)
```

This is a differential equation. The solution trajectory depends on the loss landscape geometry — specifically on the **Hessian** H = ∇²L.

Near a minimum θ*, the loss is approximately quadratic:
```
L(θ) ≈ L(θ*) + ½(θ - θ*)ᵀ H (θ - θ*)
```

The Hessian eigenvalues determine the curvature in each direction. Large eigenvalues = sharp = slow to converge stably. Small eigenvalues = flat = converge quickly.

**Gradient descent stability condition (classical):** η < 2/λ_max where λ_max is the largest Hessian eigenvalue (sharpness). Violate this → divergence.

---

## Edge of stability

Empirical finding (Cohen et al., 2021): during training with a constant learning rate η, the sharpness λ_max initially *increases* and converges to ≈ 2/η.

That's right — the model trains stably at exactly the threshold where classical theory predicts divergence.

Why? **Progressive sharpening** — the gradient descent trajectory moves toward sharper regions of the loss landscape. Then at 2/η, the dynamics become chaotic but self-stabilizing. The loss doesn't decrease monotonically anymore (it "bounces") but it still converges on a longer timescale.

**Implications for practice:**
- LR warmup gives the optimizer time to find a good trajectory before you're at full sharpness
- Gradient clipping prevents catastrophic bounces when sharpness spikes
- MuonClip (Kimi K2) applies orthogonal normalization to gradients to bound the spectral norm of weight updates, preventing loss spikes at trillion scale

---

## Implicit bias of gradient descent

Here's a fact that breaks classical intuition: **overparameterized neural networks have infinitely many global minima that achieve zero training loss**. SGD picks a specific one. Which one it picks determines generalization.

**For linear models:** gradient descent on squared loss converges to the **minimum norm** solution — the solution closest to the initialization in L2. This is a form of implicit regularization even with no explicit regularizer.

**For neural networks:** the story is more complex, but the principle holds — the optimizer's geometry (step size, noise, parameterization) acts as an implicit prior over the solution space.

**Sharpness-Aware Minimization (SAM):** makes this explicit. Instead of minimizing L(θ), minimize max_{‖ε‖≤ρ} L(θ+ε) — the loss in the *worst neighborhood* of θ. This finds flat minima that generalize better. Formally, SAM approximates ∂/∂θ [max_ε L(θ+ε)] using one gradient ascent step to find the maximizing ε.

**Grokking connection:** delayed generalization occurs because early in training, SGD finds a memorizing solution (high sharpness, overfits). Later, weight decay slowly pushes toward flatter, generalizing solutions. The "phase transition" in grokking is when the flat generalizing circuit finally dominates.

---

## Adam's geometry

Adam update for parameter θ with gradient g:
```
m ← β₁m + (1-β₁)g         # first moment (mean)
v ← β₂v + (1-β₂)g²        # second moment (variance)
m̂ = m/(1-β₁ᵗ)              # bias correction
v̂ = v/(1-β₂ᵗ)              # bias correction
θ ← θ - η · m̂/(√v̂ + ε)   # update
```

The effective learning rate for parameter i is η/√v̂ᵢ. Parameters with consistently large gradients get *smaller* effective step sizes. Parameters with small gradients get *larger* steps.

**Geometric interpretation:** Adam approximately normalizes by the diagonal of the Fisher information matrix. It's an approximation to natural gradient descent — it moves in the direction that makes the biggest change to the *output distribution*, not just the biggest change to the *parameters*.

**AdamW vs. Adam:** Adam + L2 regularization adds λθ to the gradient, making the effective weight decay depend on the adaptive step size. AdamW decouples: θ ← θ · (1 - ηλ) - η · m̂/(√v̂ + ε). The weight decay is independent of the gradient magnitude. This matters — parameters with small gradients should still decay toward zero at the same rate.

---

## μP — hyperparameter transfer across scale

**The problem:** hyperparameters (learning rate, initialization scale) that work for a 7M parameter model are completely wrong for a 70B parameter model. Tuning at 70B is prohibitively expensive.

**μP (Maximal Update Parameterization)** fixes this by choosing parameterization so that the *feature learning* dynamics are the same across width.

The key insight: in the standard parameterization, as width n → ∞:
- Hidden layer activations scale as O(1)
- Gradients w.r.t. weights scale as O(1/√n)
- Weight updates ηΔW scale as η · O(1/√n)

In the NTK parameterization (1/n weight scaling), feature updates vanish as n → ∞ → lazy training.

In **μP** (1/√n weight scaling), weight updates are O(1) regardless of width → consistent feature learning.

**Practical consequence:** tune learning rate and init scale at 7M parameters. Transfer without retuning to 7B or 70B. DeepSeek's pretraining and Kimi K2's trillion-scale training both use μP-style scaling.

---

## Muon and the optimizer frontier

**Muon (Moon et al.):** instead of normalizing by gradient variance (Adam), apply gradient descent in the *Lie group* of orthogonal matrices. The update step is:
```
G = orthogonalize(∇W)
W ← W - η · G
```

Where orthogonalize uses Newton-Schulz iterations (fast) to project the gradient onto the space of orthogonal updates.

**Why?** Adam's diagonal preconditioner doesn't account for the matrix structure of weight matrices. Muon applies updates that respect the spectral properties of the weight matrix — specifically it keeps the singular values from growing unboundedly.

**MuonClip (Kimi K2):** adds a QK-clip operation that bounds the maximum singular value of QKᵀ products. This prevents the attention entropy collapse (attention over-sharply focusing on one token) that causes loss spikes at trillion-token scale. Result: zero loss spikes over 15.5T tokens.

**The Muon claim:** ~2× compute-efficiency over AdamW at scale (same loss in half the steps). Still being validated at frontier scale; the Moonshot/Kimi result is the strongest evidence so far.

---

## Loss landscape geometry — what it looks like

Modern neural network loss landscapes are:
- **High-dimensional** (millions to billions of parameters)
- **Non-convex** (many saddle points, local minima)
- **Surprisingly flat** at good minima — most directions have near-zero curvature
- **Not as hostile as feared** — in overparameterized models, most critical points are saddle points, not local minima. Gradient descent naturally escapes them via noise.

**The double descent picture:**
1. Classical regime (underparameterized): test error decreases as model grows
2. Interpolation threshold (parameters ≈ training points): test error peaks
3. Overparameterized regime: test error *decreases again* as model grows further

In region 3, the model memorizes training data but still generalizes because it finds the minimum-norm interpolating solution. This regime is where all frontier models live.

---

## Key identities to know cold

| Identity | Context |
|---------|---------|
| η < 2/λ_max → stable | When classical GD converges |
| λ_max → 2/η at edge of stability | Why LR determines sharpness |
| ELBO = -KL(q\|\|p) + E[log p(x\|z)] | VAE objective — half optimization, half KL |
| Adam effective LR = η/√v̂ | Per-parameter adaptive step size |
| Natural gradient = F⁻¹∇L | Steepest ascent in KL geometry |
| μP: scale LR as 1/√n (width) | Hyperparameter transfer across scale |
