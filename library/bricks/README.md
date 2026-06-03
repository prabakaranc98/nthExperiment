# Bricks

Atomic quick-reference cards for recurring concepts across math, theory, algorithms, systems, and methods. One concept per brick — the definition, the key formula, where it appears in frontier work, and the common mistake.

**Use when:** you keep hitting the same concept in different papers and want a single place to look it up cold.

**Not a tutorial.** Bricks are dense reference cards, not explanations. For explanations → `library/foundations/`. For paper-level discussion → `library/reads-and-references/`.

**The full landscape:** [topic-map.md](topic-map.md) — 331 recurrent frontier ML/DL/AI concepts (the brick backlog). This index covers the **57** written so far. Bricks link to each other with `[[wiki-links]]` — install the [Foam extension](../../.vscode/extensions.json) for clickable links, backlinks, and a graph view.

---

## Index

### Math & Foundations

| Brick | One-liner |
|-------|-----------|
| [Softmax](softmax.md) | Converts a vector of real-valued logits to a probability distribution; temperature τ controls the sharpness. |
| [Layer Normalization](layer-norm.md) | Normalize each token's activation vector over the feature dimension to zero mean / unit variance, then apply a learned per-feature scale γ and shift β —… |
| [Batch Normalization](batch-norm.md) | Normalize each activation channel across the batch dimension using mini-batch mean/variance during training and a running EMA at inference; the historical… |
| [RMSNorm](rmsnorm.md) | Normalize each activation vector by its root-mean-square only — no mean subtraction, usually no bias, learned per-channel scale gain g — a cheaper… |
| [KL Divergence](kl-divergence.md) | Asymmetric measure of how different distribution Q is from distribution P; appears in RLHF, VAEs, conformal prediction, and generalization bounds. |
| [Cross-Entropy](cross-entropy.md) | Expected negative log-likelihood of data under a model's predicted distribution; the loss minimized by classification and next-token prediction; equals… |
| [Log-Derivative Trick (REINFORCE / Score Function)](log-derivative-trick.md) | ∇θ E_{x∼p_θ}[f(x)] = E_{x∼p_θ}[f(x)·∇θ log p_θ(x)] — turns the gradient of an expectation into an expectation of a gradient you can Monte Carlo estimate,… |
| [Reparameterization Trick](reparameterization.md) | Move the randomness of a sample out of the learnable parameters into fixed external noise — z = μ + σ⊙ε with ε∼N(0,I) — so ∇θ E[f(z)] becomes an… |
| [Matrix Rank](matrix-rank.md) | The number of linearly independent rows/columns of a matrix = its effective dimensionality = the count of nonzero singular values; the whole reason… |
| [SVD & Low-Rank Approximation](svd.md) | Every matrix factors as M = UΣVᵀ (orthonormal U, V; nonnegative diagonal Σ); truncating to the top-k singular values gives the *provably optimal* rank-k… |
| [Eigendecomposition](eigendecomposition.md) | A = QΛQ⁻¹ factors a square matrix into its invariant directions (eigenvectors, columns of Q) and how much it stretches each (eigenvalues, diagonal of Λ);… |
| [Jacobian](jacobian.md) | The matrix of all first-order partials J_{ij} = ∂f_i/∂x_j of a vector map f: ℝⁿ → ℝᵐ; it is the best local linear approximation of f, and the chain rule… |
| [Backpropagation / Reverse-Mode Autodiff](backpropagation.md) | The chain rule run backward through a computation graph — one forward pass caches activations, one backward pass propagates the loss gradient to every… |
| [Multilayer Perceptron (MLP)](mlp.md) | Stacked affine maps interleaved with pointwise nonlinearities — the universal-approximator primitive and the position-wise FFN that holds most of a… |
| [Byte-Pair Encoding (BPE)](bpe.md) | A greedy tokenizer-training algorithm that starts from bytes/characters and iteratively merges the most frequent adjacent symbol pair into a new token,… |

### Learning Theory & Generalization

| Brick | One-liner |
|-------|-----------|
| [Bias–Variance Tradeoff](bias-variance.md) | Expected test error decomposes as Bias² + Variance + irreducible Noise; the classical claim that capacity trades bias for variance (the U-shape) breaks in… |
| [Double Descent](double-descent.md) | Test error falls, then rises to a peak exactly at the interpolation threshold (params ≈ training points), then falls again as the model grows further… |
| [PAC-Bayes](pac-bayes.md) | Generalization bounds on the *expected* risk of a stochastic (Gibbs) classifier, controlled by KL(Q‖P) between a learned posterior Q and a… |
| [Neural Tangent Kernel (NTK)](ntk.md) | In the infinite-width limit with the right parametrization, the network stays near its initialization and training becomes kernel regression with a fixed… |
| [Edge of Stability](edge-of-stability.md) | During full-batch gradient descent, the top Hessian eigenvalue (sharpness) climbs to ≈2/η and then hovers just above it — training stably reduces loss… |
| [Implicit Bias of SGD](implicit-bias.md) | Among the infinitely many parameter settings that perfectly fit the data, GD/SGD doesn't pick one at random — it converges to structured, low-complexity… |
| [Grokking](grokking.md) | A delayed generalization phenomenon where test accuracy jumps from chance to near-perfect long after training accuracy has saturated at 100% — a sharp… |
| [Information Bottleneck](information-bottleneck.md) | Learn a representation T that maximally compresses X (min I(X;T)) while retaining everything predictive of Y (max I(T;Y)); the trade-off curve is the… |
| [Scaling Laws](scaling-laws.md) | Loss follows power laws in model size (N), dataset size (D), and compute (C = 6ND); Chinchilla says train a smaller model on more data; 2024+ labs… |
| [In-Context Learning](in-context-learning.md) | LLMs solve new tasks from prompt examples (x₁,y₁,…,x_k,y_k, x_query) without any weight update — interpretable as implicit Bayesian inference over a… |

### Architecture & Attention

| Brick | One-liner |
|-------|-----------|
| [RoPE (Rotary Position Embedding)](rope.md) | Inject position by rotating each Q/K vector's 2D feature pairs by an angle proportional to absolute position m, so the dot product qₘ·kₙ depends only on… |
| [Grouped-Query Attention (GQA / MQA)](gqa.md) | Share K/V projections across groups of query heads — MQA uses one shared K/V head, GQA uses G groups — shrinking the KV cache by H/G× at modest quality… |
| [Mixture-of-Experts Routing](moe-routing.md) | A learned router sends each token to a top-k subset of expert FFNs, decoupling total parameters from per-token FLOPs (sparse activation); the hard part is… |
| [FlashAttention](flash-attention.md) | IO-aware exact attention via tiling — computes the same result as standard attention but reads/writes HBM O(N) times instead of O(N²), giving 2-4×… |

### Training & Optimization

| Brick | One-liner |
|-------|-----------|
| [Training Stability](training-stability.md) | The set of tricks that keep large-scale (especially low-precision, high-LR) training from diverging — LR warmup, gradient clipping, QK-norm, z-loss,… |
| [LoRA — Low-Rank Adaptation](lora.md) | Fine-tune only a low-rank decomposition ΔW = AB (r << min(d,k)); freeze the pretrained weights; dramatically reduces trainable parameters. |

### Post-Training & RL

| Brick | One-liner |
|-------|-----------|
| [GRPO — Group Relative Policy Optimization](grpo.md) | Critic-free RL for LLMs — estimate the advantage baseline from a group of sampled responses instead of a value network; combined with verifiable rewards,… |
| [Direct Preference Optimization (DPO)](dpo.md) | Closed-form preference training — fit the policy directly on preference pairs with a binary-classification loss whose implicit reward is the policy's own… |
| [RLHF](rlhf.md) | Align a model to human preferences via three stages — SFT, then a reward model trained on pairwise comparisons, then PPO that maximizes reward minus a… |
| [Chain-of-Thought & Test-Time Compute](chain-of-thought.md) | Elicit intermediate reasoning tokens before the answer, then spend more inference compute (longer chains, sampling, search, verification) to trade tokens… |

### Generative Models

| Brick | One-liner |
|-------|-----------|
| [Flow Matching](flow-matching.md) | Simulation-free training of continuous normalizing flows by regressing a velocity field that transports noise to data along a prescribed probability path;… |
| [DDPM (Denoising Diffusion)](ddpm.md) | A forward process gradually corrupts data into Gaussian noise over T steps; a network is trained to predict that noise, and sampling reverses the… |

### Systems & Efficiency

| Brick | One-liner |
|-------|-----------|
| [Roofline Model](roofline.md) | Every GPU operation is either compute-bound or memory-bandwidth-bound; the roofline tells you which, and therefore how to optimize it. |
| [KV Cache](kv-cache.md) | Cache the K,V projections of all past tokens so each autoregressive decode step is O(1) in sequence length instead of O(N); the cache grows linearly with… |
| [Speculative Decoding](speculative-decoding.md) | A cheap draft model proposes k tokens autoregressively; the expensive target model verifies all k in one parallel forward pass and accepts a prefix via a… |
| [Quantization](quantization.md) | Store/compute weights and activations in low-bit formats (INT8/INT4/FP8) to cut memory and boost throughput; PTQ (cheap, post-hoc) vs QAT (trains through… |
| [Gradient Checkpointing](gradient-checkpointing.md) | Recompute activations during the backward pass instead of storing them — trade ~1 extra forward pass for an O(√L) (or better) reduction in activation… |
| [Mixed Precision](mixed-precision.md) | Run compute in 16-bit (FP16/BF16) for speed and memory while keeping an FP32 master copy of weights — and for FP16, loss scaling — to preserve numerical… |
| [ZeRO](zero.md) | Zero Redundancy Optimizer — shard the optimizer state (stage 1), gradients (stage 2), and parameters (stage 3) across data-parallel ranks instead of… |
| [FSDP — Fully Sharded Data Parallel](fsdp.md) | PyTorch's native ZeRO-3 — shards params, grads, and optimizer state across N ranks, all-gathers each layer's full weights just-in-time for compute, then… |
| [Tensor Parallelism](tensor-parallel.md) | Intra-layer model parallelism (Megatron-LM): shard individual weight matrices across GPUs so each device computes a slice of every layer, fused with… |
| [Distributed Training (Parallelism Overview)](distributed-training.md) | Frontier training composes orthogonal parallelism axes — data, tensor, pipeline, sequence/context, and expert — into a 3D/4D mesh, each axis trading a… |
| [Inference & Serving](inference-and-serving.md) | The decode-time stack — compute-bound *prefill* followed by memory-bound *decode*, scheduled via continuous batching over a paged KV cache with prefix… |

### Statistics, Evals & Uncertainty

| Brick | One-liner |
|-------|-----------|
| [Conformal Prediction](conformal.md) | Distribution-free, finite-sample prediction sets with a guaranteed coverage probability — valid under exchangeability only, no assumptions about the model… |
| [Prediction-Powered Inference (PPI)](ppi.md) | Estimate a population quantity (mean, regression coefficient, quantile) using a large pool of ML predictions, then debias with a small gold-labeled set —… |
| [Calibration & ECE](calibration.md) | A model is calibrated when its predicted confidence equals empirical accuracy (of all predictions at 80% confidence, 80% are correct); Expected… |

### Interpretability & Safety

| Brick | One-liner |
|-------|-----------|
| [Sparse Autoencoders (SAEs)](sparse-autoencoders.md) | Learn an overcomplete, sparsely-activating dictionary over model activations to decompose superposed features into (approximately) monosemantic directions… |
| [Alignment (Overview)](alignment.md) | The problem of making capable models reliably pursue intended goals/values — in 2024-26 practice a pipeline of preference learning (RLHF/DPO), AI-feedback… |
| [Safety & Dangerous-Capability Evals](safety-evals.md) | Empirical measurement of risky *capabilities* (CBRN/bio uplift, cyber-offense, autonomous replication, deception) and *propensities* (jailbreak… |

### Causal ML

| Brick | One-liner |
|-------|-----------|
| [Potential Outcomes (Neyman–Rubin)](potential-outcomes.md) | Define causal effects as contrasts of per-unit counterfactual outcomes Y(1),Y(0); only one is ever observed (the "fundamental problem of causal… |
| [DAGs & Do-Calculus](do-calculus.md) | Pearl's framework — a causal DAG plus three rewrite rules that turn an interventional query P(Y / do(X)) into estimable observational quantities; the… |
| [Causal Representation Learning (CRL)](causal-representation-learning.md) | Recover latent causal variables z and their causal graph G from high-dimensional observations x = g(z) (pixels, text), so the learned representation… |

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

File name: `kebab-case.md`. Add a row to the index above, and (optionally) tick the concept off in [topic-map.md](topic-map.md).
