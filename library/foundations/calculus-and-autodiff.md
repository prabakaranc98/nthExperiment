# Calculus & Automatic Differentiation

*Backprop, score functions, and the machinery behind gradients in frontier work.*

---

## Why this matters at the frontier

- **Backpropagation** is automatic differentiation applied to a computation graph. Understanding the chain rule in vector/matrix form is required to derive or debug any gradient-based training procedure.
- **Score functions** (∇ₓ log p(x)) appear in diffusion models, score matching, and the REINFORCE policy gradient estimator — three different fields, same mathematical object.
- **Jacobians and the chain rule** are unavoidable when reasoning about how errors propagate through transformers, why residual connections help, and how gradient checkpointing trades compute for memory.

---

## The chain rule — the foundation of backprop

**Scalar case:** if L = f(g(x)), then dL/dx = (df/dg) · (dg/dx).

**Vector case:** if L = f(y) and y = g(x), then ∂L/∂x = Jᵍ(x)ᵀ · ∂L/∂y

where Jᵍ(x) is the Jacobian matrix of g at x: Jᵢⱼ = ∂gᵢ/∂xⱼ.

This is the vector-form chain rule. Backpropagation is just this rule applied recursively through the layers of a neural network, accumulating ∂L/∂W for every weight matrix W.

---

## Backpropagation as reverse-mode automatic differentiation

Consider a computation graph: x → h₁ → h₂ → ... → L

**Forward pass:** compute each hᵢ and store them (or recompute them — see gradient checkpointing).

**Backward pass (reverse-mode AD):**
- Start with ∂L/∂L = 1
- At each node, compute local Jacobian and multiply: ∂L/∂hᵢ = Jᵀ · ∂L/∂hᵢ₊₁
- Accumulate ∂L/∂W at each weight matrix

**Computational cost:** O(1) times the cost of the forward pass. This is why backprop is efficient — one forward + one backward ≈ 2-3× a single forward pass.

**Gradient checkpointing:** instead of storing all intermediate activations (memory = O(depth)), recompute them during the backward pass (memory = O(1) per layer, at cost of extra compute). Essential for training very deep models or long context.

---

## Key gradient identities (matrix form)

These come up constantly in deriving paper equations:

| Operation | Forward | Gradient |
|-----------|---------|---------|
| y = Wx | y = Wx | ∂L/∂W = (∂L/∂y)xᵀ; ∂L/∂x = Wᵀ(∂L/∂y) |
| y = xᵀWx | scalar | ∂L/∂W = x xᵀ; ∂L/∂x = (W + Wᵀ)x |
| L = ‖y - ŷ‖² | scalar | ∂L/∂ŷ = -2(y - ŷ) |
| y = softmax(x) | vector | ∂L/∂xᵢ = yᵢ(∂L/∂yᵢ - Σⱼ yⱼ ∂L/∂yⱼ) |
| y = LayerNorm(x) | vector | Involves Gram-Schmidt-like structure |

**For attention:**
- A = softmax(QKᵀ/√d), O = AV
- ∂L/∂Q = (∂L/∂A · Kᵀ)/√d after softmax backward
- ∂L/∂K = (Qᵀ · ∂L/∂A)/√d
- ∂L/∂V = Aᵀ · ∂L/∂O

This is why FlashAttention needs to recompute A during the backward pass — it avoids storing the full N×N attention matrix but needs it for the backward.

---

## Score functions — ∇ₓ log p(x)

The **score function** is the gradient of the log-density: s(x) = ∇ₓ log p(x).

**Why gradients of log-density?**
- Taking gradient of log p instead of p avoids dealing with the normalizing constant Z (since log Z is constant w.r.t. x)
- Stein's identity: E[∇ₓ log p(x) · f(x) + ∇ₓ f(x)] = 0 for any f — useful for score estimation

**Score matching (Hyvärinen):** you can estimate the score function without knowing the normalizing constant:
```
min E[‖s_θ(x) - ∇ₓ log p(x)‖²] = min E[tr(∂s_θ/∂x) + ½‖s_θ(x)‖²] + const
```

The second form doesn't require ∇ₓ log p(x) to evaluate — just the divergence of s_θ.

**DDPM connection:** adding noise to data at level t gives p_t(x_t|x_0). The score ∇_{x_t} log p_t(x_t) points toward less noisy data. DDPM trains ε_θ(x_t, t) ≈ noise, which is equivalent (up to scaling) to training s_θ(x_t, t) ≈ ∇_{x_t} log p_t(x_t). The denoising objective IS score matching.

---

## REINFORCE and the policy gradient

In RL, the objective is J(θ) = E_{τ~π_θ}[R(τ)] where τ is a trajectory and R(τ) its return.

Taking the gradient:
```
∇_θ J(θ) = E[∇_θ log π_θ(τ) · R(τ)]
         = E[Σₜ ∇_θ log π_θ(aₜ|sₜ) · R(τ)]
```

This is the **score function estimator** (also called REINFORCE). Key step:
∇_θ E_{x~p_θ}[f(x)] = E_{x~p_θ}[∇_θ log p_θ(x) · f(x)]

This identity (derived via log-derivative trick: ∇p = p·∇log p) lets you take gradients through expectations even when you can't backprop through the sampling step.

**The same trick appears in:**
- Policy gradient / REINFORCE in RL
- NVIL / REINFORCE estimator for discrete latent variables in VAEs
- Score-based generative models

**High variance problem:** the REINFORCE estimator is unbiased but has high variance. Solution: subtract a baseline b(s) from R(τ): ∇J = E[∇log π · (R - b)]. Doesn't change the mean (baseline has zero expected gradient) but reduces variance dramatically. GRPO uses group mean as the baseline.

---

## Automatic differentiation in practice

Modern frameworks (PyTorch, JAX) implement **reverse-mode AD** via a computation graph built during the forward pass.

**PyTorch:** dynamic graph (define-by-run). Each operation records what created it. `.backward()` traverses the graph in reverse.

**JAX:** functional, stateless. `jax.grad(f)(x)` returns the gradient function. `jax.jit` compiles to XLA. `jax.vmap` vectorizes over batches. Everything is a pure function.

**Key JAX patterns:**
- `grad(f)` → scalar-to-scalar gradient
- `jacfwd(f)` / `jacrev(f)` → full Jacobian (forward/reverse mode)
- `hessian(f) = jacfwd(jacrev(f))` → Hessian
- `jit` + `vmap` + `grad` compose cleanly

**When to use forward vs. reverse mode:**
- Reverse mode (backprop): efficient when outputs << inputs (most ML: scalar loss, many parameters)
- Forward mode: efficient when inputs << outputs (scientific computing, sensitivity analysis)
