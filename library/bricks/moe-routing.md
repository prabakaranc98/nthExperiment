# Mixture-of-Experts Routing

**One-liner:** A learned router sends each token to a top-k subset of expert FFNs, decoupling total parameters from per-token FLOPs (sparse activation); the hard part is keeping experts balanced so none idle or overflow.

## The mechanism

Replace a dense FFN with E expert FFNs and a router (linear gate) Wᵣ. For token x:

  g = softmax(x·Wᵣ)        ∈ ℝᴱ      (router logits → gate probs)
  S = top-k(g)             (indices of k largest gates, usually k=1 or 2)
  y = Σ_{i∈S} (gᵢ / Σ_{j∈S} gⱼ) · Eᵢ(x)

Only k of E experts run per token, so per-token compute ≈ dense FFN × k, while parameter count scales with E. "Sparse upcycling" inits experts from a dense checkpoint.

## Load balancing (the actual problem)

Routing is discrete → no gradient through top-k, and the router collapses to a few favorite experts. Two fixes, usually combined:

- **Auxiliary loss** (Switch/GShard): L_aux = α·E·Σᵢ fᵢ·Pᵢ, where fᵢ = fraction of tokens routed to i, Pᵢ = mean gate prob for i. Minimized when load is uniform. α ≈ 0.01.
- **Capacity factor:** each expert holds ≤ C = cf·(tokens/E) tokens; overflow tokens are *dropped* (skip the FFN via residual). cf ≈ 1.0–1.25.
- **DeepSeek-V3 (2024):** *auxiliary-loss-free* balancing — add a per-expert bias bᵢ to routing logits, adjusted online toward balance, so the main loss isn't perturbed.

## Where it appears

- **Switch Transformer / GShard** — top-1 / top-2 token-choice routing, the canonical formulation.
- **Mixtral 8x7B** — 8 experts, top-2, ~47B total params but ~13B active per token.
- **DeepSeek-V3 / V2** — fine-grained experts + shared (always-on) experts + aux-loss-free balancing; 671B total, 37B active.
- **GLaM, Qwen-MoE, DBRX, Grok-1** — production sparse LLMs trading memory/bandwidth for cheap FLOPs.
- **Expert-choice routing** — flip it: each expert picks its top tokens, giving perfect balance by construction (but breaks causality/streaming).

## Common mistake

Thinking MoE saves memory or makes serving cheap. It saves *FLOPs per token*, not parameters — all E experts must sit in VRAM, and routing scatters tokens across experts/devices, making it bandwidth- and all-to-all-communication-bound. The win is compute efficiency at fixed quality, not a smaller footprint.

## See also
- [[mlp]] — each expert is a standard FFN/MLP block; MoE swaps one dense MLP for many sparse ones
- [[softmax]] — the router gate is a softmax over experts; top-k truncates it
- [[tensor-parallel]] — expert parallelism shards experts across devices, needing all-to-all dispatch
