# Training Stability

**One-liner:** The set of tricks that keep large-scale (especially low-precision, high-LR) training from diverging — LR warmup, gradient clipping, QK-norm, z-loss, careful init, and loss-spike recovery — all aimed at bounding activation/logit/gradient magnitudes so updates stay well-conditioned.

## The failure modes

Divergence = loss spikes to NaN/Inf or plateaus high. Root causes:
- **Attention logit blow-up:** QKᵀ/√d grows unbounded → softmax saturates → near-zero grad. Worst in bf16/fp8.
- **Output logit drift:** unnormalized logits grow → softmax overflow, large z = log Σ exp(zᵢ).
- **Init/LR mismatch:** large updates early when params are mis-scaled → irrecoverable spike.
- **Bad data batch + high LR:** a single rare batch produces a huge gradient.

## The standard toolkit

**LR warmup:** ramp 0 → η_max over ~0.5–2k steps. Early Adam second-moment estimates are noisy; warmup avoids huge initial steps.

**Gradient clipping (global L2 norm):**
g ← g · min(1, c / ‖g‖₂), typically c ∈ [0.5, 1.0]. Clips the *whole* grad vector, preserving direction.

**QK-norm:** RMSNorm/L2-normalize Q and K before the dot product → bounds attention logits, removes the dominant fp8/bf16 spike source. (Used in many 2024–25 frontier models.)

**z-loss:** add β · (log Σ exp(zᵢ))² to the loss (β ≈ 1e-4). Pushes the softmax normalizer toward 0, keeping logits small and numerically tame.

**Careful init:** scale residual-branch outputs by 1/√(2·n_layers) (GPT-2 style) so residual variance doesn't grow with depth; μP transfers stable HPs across width.

**Loss-spike recovery:** checkpoint frequently; on spike, restore last good checkpoint, skip the offending batches, optionally lower LR briefly.

## Where it appears

- **PaLM (Chowdhery 2022)** — documented spikes; mitigated by restart-from-checkpoint + batch-skipping.
- **OLMo / Gemma / DeepSeek-V3** — QK-norm + z-loss + warmup as standard recipe; DeepSeek-V3 reported a fully spike-free fp8 run.
- **GPT-2 / GPT-3** — residual-scaled init, grad clip at 1.0, warmup.
- **μTransfer (Yang 2021)** — stability via principled init/LR scaling across width.

## Common mistake

Treating loss spikes as purely a learning-rate problem. Lowering LR often just delays the spike; the real fix is usually bounding the quantity that blows up (QK-norm for attention logits, z-loss for output logits) and using bf16 master weights / fp32 optimizer state. Also: per-parameter clipping (clipping each coordinate) breaks gradient direction — use *global-norm* clipping.

## See also
- [[mixed-precision]] — most instabilities surface only in bf16/fp8; loss scaling and fp32 master weights are part of the cure
- [[rmsnorm]] — the normalizer behind QK-norm and modern pre-norm transformers
- [[edge-of-stability]] — the sharpness/LR dynamics that govern when training sits on the brink of divergence
