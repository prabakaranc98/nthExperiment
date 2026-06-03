# Residual Stream

**One-liner:** The additive skip-connection backbone of a transformer reinterpreted as a single high-dimensional vector space that every block reads from and writes to — a shared linear "communication channel" running from embedding to unembedding, and the central object of mechanistic interpretability.

## The key insight

A pre-norm transformer's forward pass is purely additive at the top level:

x₀ = embed(tokens) + pos
xₗ = xₗ₋₁ + Attnₗ(LN(xₗ₋₁))
xₗ = xₗ + MLPₗ(LN(xₗ))
logits = Unembed(LN(x_L))

So the residual stream xₗ is the **running sum** of every component's output: x_L = embed + Σ (head and MLP contributions). Each block reads its input via LN, computes a delta, and **adds** it back — no component overwrites the stream, it only writes into it. Because addition is linear, you can decompose any later activation (or logit) into a sum of contributions from earlier components and trace information flow. Components communicate by writing to and reading from (often orthogonal/low-rank) subspaces of the same d_model space; "memory management" heads can even subtract to delete information.

## Where it appears

- **Mathematical Framework for Transformer Circuits (Elhage et al., 2021)** — coins "residual stream"; QK/OV circuits, virtual weights, and head composition are all framed as reads/writes to it.
- **Logit lens / tuned lens** — apply the unembedding to intermediate xₗ, exploiting that the stream is in (roughly) the same basis throughout depth.
- **Activation patching, steering vectors, SAEs** — all operate by editing or decomposing the residual stream at a chosen layer; SAEs are trained directly on stream activations.
- **Pre-norm architecture (GPT-2 onward, all frontier LLMs)** — the clean additive stream exists *because* normalization is moved inside the branch, leaving the skip path identity.

## Common mistake

Thinking the residual stream has a single fixed semantic basis. Individual dimensions are not interpretable features (superposition); directions are, but there is no privileged orthogonal basis, and LayerNorm/RMSNorm rescaling plus a growing stream norm with depth means raw magnitudes across layers aren't directly comparable.

## See also
- [[residual-skip-connections]] — the architectural primitive the stream is built from
- [[pre-norm-vs-post-norm-vs-sandwich-norm]] — pre-norm is what makes the additive stream clean
- [[superposition]] — why stream directions, not neurons, carry features
