# Initialization & Residual/Depth Scaling

**One-liner:** Choose weight-init variances and downscale residual branches (typically by 1/√(2L) or per-layer) so that at step zero forward activation variance and backward gradient norms stay O(1) across depth, making very deep nets trainable without warmup tricks or norm placement hacks.

## The formula / definition

**Init variance (fan-in / He / Glorot):** for a linear layer y = Wx, pick Var(W) so Var(y) ≈ Var(x). He: Var(W) = 2/fan_in (ReLU); Glorot: 2/(fan_in+fan_out). This keeps a single layer variance-preserving — but residual *stacking* still grows variance.

**The residual problem.** With pre-norm blocks x_{l+1} = x_l + f(x_l), variance accumulates: Var(x_L) ≈ Var(x_0) + Σ Var(f_l) grows ~linearly in depth L. The residual stream norm blows up with depth, shrinking the relative contribution of each block.

**Fixes (downscale the residual branch):**
- **1/√depth scaling:** x_{l+1} = x_l + (1/√(2L))·f(x_l). Keeps total variance O(1).
- **Fixup (Zhang et al. 2019):** norm-free deep resnets — scale init of the last layer in each branch to 0 (or by L^{-1/(2m-2)} for m layers/branch), zero-init final branch layer, add learnable scalar biases. Trains 10k-layer nets with no normalization.
- **DeepNet / DEEPNORM (2022):** post-norm with x_{l+1} = LN(α·x_l + f(x_l)), α = (2N)^{1/4} for N-layer encoder; downscale select weight inits by β = (8N)^{-1/4}. Bounds the update magnitude → stable 1000-layer Transformers.
- **ReZero:** x_{l+1} = x_l + α_l·f(x_l) with learnable α_l initialized to 0 — the network starts as identity and gates branches in gradually.
- **SkipInit / LayerScale:** per-channel learnable γ (init ~1e-4 to 1e-6) on each residual branch (used in CaiT, ViT-22B).

## Where it appears

- **GPT-2 / GPT-3** — scale residual-projection init by 1/√(num_layers) (the canonical 1/√N residual trick).
- **DeepNet / Foundation Transformers** — DEEPNORM enables 1000-layer encoder-decoders.
- **ViT-22B, CaiT, large diffusion U-Nets/DiTs** — LayerScale with tiny init to stabilize deep/wide vision transformers.
- **μP (maximal update parameterization)** — depth/width-aware init+LR scaling so HPs transfer across scale; complements residual downscaling.

## Common mistake

Conflating per-layer init (He/Glorot makes *one* layer variance-preserving) with depth stability. Even perfect per-layer init lets residual-stream variance grow ~linearly in L; you also need branch downscaling (1/√L, LayerScale, or DeepNorm). The other classic error: zero-initializing the *whole* branch instead of just its last layer — that kills the gradient signal the branch needs to learn.

## See also
- [[residual-skip-connections]] — the architecture these scalings are designed to stabilize
- [[pre-norm-vs-post-norm-vs-sandwich-norm]] — norm placement interacts directly with residual variance growth
- [[maximal-update-parameterization]] — width/depth-aware init+LR scaling for HP transfer
