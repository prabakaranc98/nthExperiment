# Bricks

Atomic quick-reference cards for recurring concepts across math, theory, algorithms, systems, and methods. One concept per brick — the definition, the key formula, where it appears in frontier work, and the common mistake.

**Use when:** you keep hitting the same concept in different papers and want a single place to look it up cold.

**Not a tutorial.** Bricks are dense reference cards, not explanations. For explanations → `library/foundations/`. For paper-level discussion → `library/reads-and-references/`.

---

## Index

### Math

| Brick | One-liner |
|-------|-----------|
| [softmax.md](softmax.md) | Normalizes logits to a probability distribution; temperature scales the sharpness |
| [layer-norm.md](layer-norm.md) | Normalizes over the feature dimension; what transformers actually use |
| [kl-divergence.md](kl-divergence.md) | Asymmetric divergence between distributions; in RLHF, VAEs, conformal prediction |
| [cross-entropy.md](cross-entropy.md) | Training loss = negative log-likelihood = H(P,Q) |
| [log-derivative-trick.md](log-derivative-trick.md) | ∇E[f(x)] = E[f(x)∇log p(x)] — the REINFORCE / score function estimator |
| [reparameterization.md](reparameterization.md) | z = μ + σε; makes sampling differentiable (VAEs, flow matching) |
| [matrix-rank.md](matrix-rank.md) | Number of independent rows/columns; LoRA, MLA, NTK |
| [svd.md](svd.md) | M = UΣVᵀ; low-rank approximation; LoRA motivation |
| [eigendecomposition.md](eigendecomposition.md) | A = QΛQᵀ; loss curvature, NTK, SAM sharpness |
| [jacobian.md](jacobian.md) | ∂f/∂x matrix; backprop chain rule in vector form |

### Theory

| Brick | One-liner |
|-------|-----------|
| [bias-variance.md](bias-variance.md) | E[L] = Bias² + Variance + Noise; breaks in the overparameterized regime |
| [double-descent.md](double-descent.md) | Test error falls again past the interpolation threshold |
| [pac-bayes.md](pac-bayes.md) | Generalization bounds via KL(Q‖P); non-vacuous for modern nets |
| [ntk.md](ntk.md) | Infinite-width limit → fixed kernel; lazy vs. feature-learning regimes |
| [edge-of-stability.md](edge-of-stability.md) | λ_max → 2/η during training; why LR determines sharpness |
| [implicit-bias.md](implicit-bias.md) | SGD finds minimum-norm interpolating solution; the hidden regularizer |
| [grokking.md](grokking.md) | Generalization long after memorization; circuit formation phase transition |
| [scaling-laws.md](scaling-laws.md) | Loss ∝ N^α · D^β · C^γ; Chinchilla-optimal vs. inference-optimal |
| [information-bottleneck.md](information-bottleneck.md) | Compress X while preserving Y; connects to representation learning theory |

### Algorithms & Methods

| Brick | One-liner |
|-------|-----------|
| [bpe.md](bpe.md) | Byte-Pair Encoding; merge most frequent pairs iteratively |
| [flash-attention.md](flash-attention.md) | IO-aware exact attention via tiling; O(N) HBM, 2-4× faster |
| [rope.md](rope.md) | Rotary Position Embedding; relative positions via rotation in Q/K |
| [gqa.md](gqa.md) | Grouped Query Attention; KV heads shared across Q heads |
| [moe-routing.md](moe-routing.md) | Top-k gating; aux-loss-free balancing; expert specialization |
| [lora.md](lora.md) | ΔW = AB, r << min(d,k); fine-tuning in low-rank subspace |
| [grpo.md](grpo.md) | Group-relative advantage; critic-free RL for reasoning |
| [dpo.md](dpo.md) | Direct Preference Optimization; closed-form preference training without RL |
| [conformal.md](conformal.md) | Distribution-free coverage guarantee; requires only exchangeability |
| [ppi.md](ppi.md) | Prediction-Powered Inference; valid inference with ML-labeled data |
| [flow-matching.md](flow-matching.md) | Simulation-free training of continuous flows; unifies with diffusion |
| [ddpm.md](ddpm.md) | Denoising Diffusion; forward noising + reverse denoising = generative model |

### Systems

| Brick | One-liner |
|-------|-----------|
| [roofline.md](roofline.md) | Performance = min(peak FLOPS, bandwidth × arithmetic intensity) |
| [kv-cache.md](kv-cache.md) | Store K,V for past tokens; grows linearly with context; the inference bottleneck |
| [zero.md](zero.md) | ZeRO: shard optimizer state / gradients / params across ranks |
| [fsdp.md](fsdp.md) | Fully Sharded Data Parallel; PyTorch's ZeRO-3 implementation |
| [tensor-parallel.md](tensor-parallel.md) | Split weight matrices across GPUs; Megatron-LM style |
| [speculative-decoding.md](speculative-decoding.md) | Draft-and-verify; faster inference without quality loss |
| [quantization.md](quantization.md) | INT8/INT4/FP8; reduce precision → memory and throughput |
| [gradient-checkpointing.md](gradient-checkpointing.md) | Recompute activations during backward; trade compute for memory |
| [mixed-precision.md](mixed-precision.md) | FP16/BF16 forward + FP32 optimizer; standard since 2018 |

---

## How to add a brick

Copy this template:

```markdown
# [Concept Name]

**One-liner:** [single sentence definition]

## The formula / definition

[The core math or pseudocode]

## Where it appears

- [Paper / technique 1] — [how it's used]
- [Paper / technique 2] — [how it's used]

## Common mistake

[The one thing people get wrong]

## See also

- [[related-brick]]
```

File name: `kebab-case.md`. Add a row to the index above.
