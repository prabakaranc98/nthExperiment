# Backpropagation / Reverse-Mode Autodiff

**One-liner:** The chain rule run backward through a computation graph — one forward pass caches activations, one backward pass propagates the loss gradient to every parameter at total cost ~2-3× the forward pass, independent of parameter count.

## The key insight

A scalar loss L is a composition of ops forming a DAG. Each node has a local Jacobian; the gradient w.r.t. any input is the product of local Jacobians along all paths to L. Reverse mode computes this efficiently because the output is a *scalar*: seed the output cotangent with 1 and pull it back.

For a layer y = f(x), given the incoming cotangent ḡ = ∂L/∂y, the **vector-Jacobian product (VJP)** gives the outgoing one:

  ∂L/∂x = Jᶠ(x)ᵀ ḡ

No full Jacobian is ever materialized — only the VJP. For a linear layer y = Wx + b:
  ∂L/∂x = Wᵀ ḡ,  ∂L/∂W = ḡ xᵀ,  ∂L/∂b = ḡ

## Why reverse mode (not forward)

Cost of one sweep ≈ (#outputs) for forward mode, (#inputs) for reverse mode. NN training has 1 scalar output and millions of inputs (params), so reverse mode wins: **all** gradients in one backward pass. The catch: you must store every intermediate activation needed by the VJPs → O(depth) memory.

## Where it appears

- **Every NN training loop** — PyTorch `autograd`, JAX `grad`/`vjp`, TensorFlow `GradientTape` build the tape/graph and replay it in reverse.
- **Gradient checkpointing** — trade compute for memory by *not* storing all activations; recompute them in the backward pass (the FSDP/long-context training enabler).
- **Mixed precision** — backward runs in bf16/fp16 with loss scaling to keep small gradients from underflowing.
- **Adjoint methods** — Neural ODEs / continuous flows are backprop's continuous-time analogue (solve the adjoint ODE backward).

## Common mistake

Confusing backprop with SGD. Backprop only *computes* the gradient; the optimizer (SGD, Adam) *uses* it. Relatedly: thinking it's a numerical approximation — it's exact analytic differentiation (to floating-point error), not finite differences.

## See also
- [[jacobian]] — backprop is repeated vector-Jacobian products; the Jacobian is the local linear map at each node
- [[gradient-checkpointing]] — manages backprop's O(depth) activation-memory cost
- [[mixed-precision]] — the numerical regime real backward passes run in
