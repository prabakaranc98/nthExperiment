# Multilayer Perceptron (MLP)

**One-liner:** Stacked affine maps interleaved with pointwise nonlinearities — the universal-approximator primitive and the position-wise FFN that holds most of a transformer's parameters.

## The formula / definition

A depth-L MLP composes affine layers with a nonlinearity σ:

h₀ = x
hₗ = σ(Wₗ hₗ₋₁ + bₗ),  l = 1..L−1
y  = W_L h_{L−1} + b_L

Each layer is a linear map (the only place weights live) followed by an elementwise σ (ReLU, GELU, SiLU). Without σ the whole stack collapses to a single affine map — nonlinearity is what gives expressivity. **Universal approximation** (Cybenko 1989, Hornik 1991): one hidden layer with enough units approximates any continuous function on a compact set arbitrarily well — an existence result, *not* a statement about learnability, width needed, or generalization.

## The transformer FFN

Inside each transformer block, the position-wise FFN is a 2-layer MLP applied independently per token:

FFN(x) = W₂ · σ(W₁ x + b₁) + b₂,   d_model → d_ff → d_model,   d_ff ≈ 4·d_model

This is typically **most of the parameters** in an LLM. Modern variants use **gated linear units** (GLU/SwiGLU, Shazeer 2020): FFN(x) = W₂ (σ(W₁x) ⊙ V x), with d_ff shrunk to ~⅔·4·d_model to match the parameter count.

## Where it appears

- **Every transformer block** — the FFN/MLP sublayer after attention; Llama/Mistral/Qwen use SwiGLU
- **Mixture-of-Experts** — each expert is just an FFN; routing selects k of N MLPs per token (Mixtral, DeepSeek-V3)
- **Diffusion / flow models** — time- and condition-embedding MLPs, and the per-token MLP in DiT blocks
- **Heads everywhere** — projection/classification heads, value heads in RLHF, contrastive projection heads

## Common mistake

Believing universal approximation implies a *practical* network can learn the function — it is an existence theorem with no bound on width, no guarantee SGD finds the weights, and nothing about generalization. Also: stacking linear layers with no nonlinearity buys zero expressivity (it's still one affine map).

## See also
- [[backpropagation]] — how MLP weights are actually trained (reverse-mode AD through the chain rule)
- [[moe-routing]] — sparse MoE swaps the dense FFN for routed expert MLPs
- [[ntk]] — infinite-width MLPs become a fixed kernel, the theory baseline for these nets
