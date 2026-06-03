# maximal update parameterization (muP)

**One-liner:** Width-aware scaling of init variances and per-layer learning rates (Tensor Programs framework) that holds the network in the feature-learning regime as width n grows, so optimal HPs found on a tiny proxy model transfer zero-shot to the full-width run.

## The formula / definition

The scaling rules vs. width n (per layer type), relative to standard parameterization:

| | init var | LR (Adam) | LR (SGD) | output mult |
|---|---|---|---|---|
| input/embed | Θ(1) | Θ(1) | Θ(n) | — |
| hidden (n×n) | Θ(1/n) | Θ(1/n) | Θ(1) | — |
| output/readout | Θ(1/n²) | Θ(1/n) | Θ(1/n) | 1/n |

Key invariant: every layer's *activations and their updates* are Θ(1) in n. The readout is initialized small (or zeroed) and its logits scaled by 1/n so the network starts near-zero-output but still learns features (unlike NTK/lazy, where features barely move).

In practice you fix a base width n₀, tune HPs there, and multiply by width ratio m = n/n₀: hidden LR ← lr·(1/m), init keeps fan-in variance, readout multiplier ← 1/m. This is "μTransfer."

## Where it appears

- Tensor Programs IV/V (Yang & Hu 2021, Yang et al. 2022) — original derivation and μTransfer of LR, init, multipliers across width.
- GPT-4 / Cerebras-GPT / MiniCPM — tuned HPs on small proxies, transferred to the large run, saving the bulk of HP-search compute.
- nanoGPT-speedrun, modded-nanoGPT, and Muon-based recipes — μP-style per-layer LR scaling is standard practice; depth-μP extends it to scaling layers ∝ 1/√depth.

## Common mistake

Thinking μP transfers HPs across *data/tokens or depth* by default. Vanilla μP only guarantees transfer across **width** (and batch size, somewhat). Depth transfer needs Depth-μP (residual branch scaling ∝ 1/√L); transferring across token count still requires the schedule/LR to respect data-scaling. Also: μP transfers the *shape* of the loss-vs-LR curve, so the proxy must be wide enough that you're already in the asymptotic regime.

## See also
- [[ntk]] — the lazy/kernel limit μP is explicitly designed to escape
- [[feature-learning-vs-lazy-training]] — μP is the parameterization that stays in the feature-learning regime at any width
- [[hyperparameter-scaling-laws]] — the empirical sibling: fit HP optima as functions of N, D rather than deriving them
