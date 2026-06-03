# Diffusion Forcing / Block-Autoregressive Diffusion

**One-liner:** Train a sequence model to denoise each token at its own independent noise level, unifying next-token autoregression (full-noise future, clean past) and diffusion (uniform noise) as endpoints of one objective — enabling variable-horizon, stable long rollouts for video/sequences.

## The key insight

Assign every token t its own noise level k_t and corrupt it independently: x_t^{(k_t)} = α_{k_t} x_t + σ_{k_t} ε_t. Train a (usually causal) model to predict the denoising target for all tokens at once given the *per-token* noise schedule:

L = E_{t, k_t, ε} [ ‖ ε_θ(x_{1:T}^{(k_{1:T})}, k_{1:T}) − target_t ‖² ]

The noise level acts as a per-token masking knob:
- All k_t = 0 → clean (teacher forcing limit)
- Future tokens at k=max (pure noise), past at k=0 → recovers **causal autoregression** (a noised future token carries no info, like a mask)
- All k_t equal → standard (full-sequence) **diffusion**

**Sampling** = a 2D scheduling matrix over (token position × denoise step): roll a diagonal/triangular front forward, denoising newer tokens while older ones are already clean. Lets you trade compute for horizon and resample/guide partially-generated futures.

## Where it appears

- **Diffusion Forcing** (Chen et al., NeurIPS 2024) — the original formulation; RNN backbone for video prediction, maze planning, and flexible-horizon rollout with Monte Carlo Guidance.
- **Diffusion Forcing Transformer / History-Guided Video Diffusion** (Song et al., 2025) — transformer backbone, long causal video generation with history guidance over noisy context.
- **Block-wise / block-causal diffusion LMs** (e.g. Block Diffusion, BD3-LM) — diffuse within a block, autoregress across blocks; combines KV-cache reuse with parallel intra-block denoising.
- **Long-horizon world models & robot policies** — stable autoregressive video/action rollout where pure AR drifts and pure diffusion can't extend past training length.

## Common mistake

Thinking the future tokens are *masked* the way a discrete masked-diffusion model masks them. They are **continuously noised at independent levels** — a high-noise token still passes a (nearly uninformative) noisy embedding through the network, and the per-token level k_t is itself a conditioning input. The unification with AR is a limit (k→max), not literal token masking, and the schedule is a continuous 2D grid, not a binary mask pattern.

## See also
- [[ddpm]] — the per-token denoising objective each token is trained under
- [[video-generation-spatiotemporal-modeling]] — primary application: long, stable autoregressive video rollout
- [[teacher-forcing-exposure-bias]] — clean-past/noised-future recovers teacher forcing and mitigates its rollout drift
