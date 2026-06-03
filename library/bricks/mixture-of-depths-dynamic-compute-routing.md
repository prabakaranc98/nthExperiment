# Mixture-of-Depths / Dynamic Compute Routing

**One-liner:** Per-layer routers let only the top-k tokens enter each block's full compute (attention + MLP) while the rest take the residual skip, giving a fixed (not data-dependent) compute budget that spends FLOPs adaptively across the sequence depth-wise.

## The mechanism (Raposo et al., 2024)

For each MoD block, a linear router scores every token: r_i = w_r · x_i (scalar). With capacity c (e.g. 12.5% of T tokens), only the top-c tokens by router weight are processed; the rest bypass:

    x_i' = x_i + r_i · block(x_i)   if i ∈ top-k(r, c·T)
    x_i' = x_i                       otherwise

Multiplying block output by r_i puts the router on the gradient path (like MoE gating), so the scores are learnable. Because k = c·T is fixed, the FLOPs and the per-layer tensor shape are static — a key contrast to early-exit / data-dependent halting. MoD blocks are typically interleaved every other layer.

## The causality problem and fix

Top-k over a sequence is a non-causal operation (a token's selection depends on others). At train time it's fine (you see the whole sequence), but autoregressive decoding can't peek. Fix: train a small **auxiliary router/predictor** (an MLP or a per-token sigmoid "will this token be in the top-k?" classifier) so inference decides selection causally, token-by-token, matching the train-time distribution.

## Where it appears

- **Mixture-of-Depths** (DeepMind, 2024) — the original; ~50% fewer FLOPs/forward-pass at matched loss, or faster step time at iso-params.
- **MoDE / MoD+MoE** — stack depth routing on top of expert routing; tokens choose both *which* experts and *whether* to do the block at all.
- **CoLT5 / conditional computation** — earlier routed light-vs-heavy branches per token; same "spend compute where it matters" thesis.
- **Mixture-of-Recursions (2025)** — routes tokens through a recursively-shared block a variable number of times, a recurrent cousin of MoD.

## Common mistake

Conflating it with early-exit / adaptive halting. Those give a *data-dependent* compute budget (variable batch shapes, hard to batch). MoD fixes k per layer, so the budget and tensor shapes are **static and known ahead of time** — the adaptivity is in *which* tokens, not *how many*. Also: it routes over depth, not over experts, so it composes with MoE rather than replacing it.

## See also
- [[moe-routing]] — sibling primitive; same top-k gating math but routing over experts (width) instead of layers (depth)
- [[expert-load-balancing]] — both face the same router-collapse / uneven-utilization failure modes
- [[residual-skip-connections]] — the skip is exactly the "do-nothing" path a bypassed token takes
