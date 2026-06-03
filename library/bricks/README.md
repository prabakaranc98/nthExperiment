# Bricks

Atomic quick-reference cards for recurring concepts across math, theory, algorithms, systems, and methods. One concept per brick — definition, key formula, where it appears in frontier work, the common mistake.

**Use when:** you keep hitting the same concept in different papers and want a single place to look it up cold.

**Not a tutorial.** Bricks are dense reference cards. For explanations → `library/foundations/`. For paper-level discussion → `library/reads-and-references/`.

**The full landscape:** [topic-map.md](topic-map.md) — 331 recurrent frontier concepts, all now written. Bricks link to each other with `[[wiki-links]]` — install the [Foam extension](../../.vscode/extensions.json) for clickable links, backlinks, and a graph view.

---
## Index

### Transformer Architecture
| Brick | One-liner |
|-------|-----------|
| [Scaled Dot-Product Attention](scaled-dot-product-attention.md) | Softmax(QKᵀ/√dₖ)V — a value-weighted average where weights come from query-key dot products, with the √dₖ scaling holding logits in softmax's… |
| [Multi-Head Attention](multi-head-attention.md) | Run H attention ops in parallel on linearly-projected d/H-dim subspaces and concatenate, letting heads specialize on different relations; the head is… |
| [Grouped-Query Attention (GQA / MQA)](gqa.md) | Share K/V projections across groups of query heads — MQA uses one shared K/V head, GQA uses G groups — shrinking the KV cache by H/G× at modest… |
| [Multi-Head Latent Attention (MLA)](multi-head-latent-attention.md) | Compress K/V into one low-rank latent vector cached per token (with RoPE carried on a small *decoupled* key portion), shrinking the KV cache below… |
| [RoPE (Rotary Position Embedding)](rope.md) | Inject position by rotating each Q/K vector's 2D feature pairs by an angle proportional to absolute position m, so the dot product qₘ·kₙ depends only… |
| [RoPE Context Extension (NTK / YaRN / Position Interpolation)](rope-context-extension.md) | Extend a RoPE model's usable context past its training length by rescaling rotation frequencies — linear Position Interpolation squeezes all… |
| [NoPE / Positional-Scheme Length Generalization](nope-positional-scheme-length-generalization.md) | Decoder-only transformers can encode and extrapolate token order with NO explicit positional encoding — the causal mask alone breaks permutation… |
| [Pre-Norm vs Post-Norm vs Sandwich Norm](pre-norm-vs-post-norm-vs-sandwich-norm.md) | Where the normalizer sits relative to the residual add — inside the branch (Pre-Norm, identity-preserving residual stream, trains deep without… |
| [RMSNorm](rmsnorm.md) | Normalize each activation vector by its root-mean-square only — no mean subtraction, usually no bias, learned per-channel scale gain g — a cheaper… |
| [QK-Normalization](qk-normalization.md) | Apply a normalization (typically RMSNorm/L2) to queries and keys *before* the dot product so attention logits stay bounded — preventing the… |
| [Residual Stream](residual-stream.md) | The additive skip-connection backbone of a transformer reinterpreted as a single high-dimensional vector space that every block reads from and writes… |
| [SwiGLU / GeGLU](swiglu-geglu.md) | Gated FFN variants that split the up-projection into a gate branch (passed through Swish/GELU) and a value branch, multiply them elementwise, then… |
| [Attention Sinks](attention-sinks.md) | Softmax forces attention weights to sum to 1, so when no token is relevant, heads dump excess mass onto the first token(s) — these "sinks" are… |
| [Sliding-Window / Local Attention](sliding-window-local-attention.md) | Restrict each query to attend only to the previous w keys (a fixed window), cutting attention cost to O(n·w) and capping KV memory at w per layer,… |
| [Mixture-of-Experts Routing](moe-routing.md) | A learned router sends each token to a top-k subset of expert FFNs, decoupling total parameters from per-token FLOPs (sparse activation); the hard… |
| [Expert Load Balancing](expert-load-balancing.md) | Mechanisms (auxiliary load-balancing loss, or aux-loss-free per-expert bias correction) plus capacity-factor / token-dropping tuning that stop an MoE… |
| [GELU / SiLU (Smooth Activations)](gelu-silu.md) | Smooth, non-monotonic gating-style nonlinearities — GELU multiplies x by its Gaussian CDF, SiLU/Swish multiplies x by its sigmoid — used as the base… |
| [Residual / Skip Connections](residual-skip-connections.md) | y = x + F(x); an identity shortcut around a block so the layer learns a residual rather than a full transform — this makes 100+ layer nets trainable… |
| [Softmax Bottleneck & Logit Cap / Final-Layer Tying](softmax-bottleneck-logit-cap-final-layer-tying.md) | A softmax over logits W·h is rank-limited by hidden dim d (the "softmax bottleneck"), and the output matrix W is often *tied* to the input embedding… |
| [Cosine / Temperature in Contrastive & Attention Logits](cosine-temperature-in-contrastive-attention.md) | A scalar divisor applied to dot-product logits before softmax/sigmoid — fixed 1/√d_k in attention to control variance, learned (often capped) in… |
| [Multi-Token Prediction (MTP)](multi-token-prediction.md) | Predict the next *k* tokens at each position instead of just one — used as an auxiliary training objective for denser supervision (DeepSeek-V3)… |
| [Mixture-of-Depths / Dynamic Compute Routing](mixture-of-depths-dynamic-compute-routing.md) | Per-layer routers let only the top-k tokens enter each block's full compute (attention + MLP) while the rest take the residual skip, giving a fixed… |

### Attention Alternatives & Sequence Models
| Brick | One-liner |
|-------|-----------|
| [Linear Attention (Kernel-Feature / Associative Form)](linear-attention.md) | Replace softmax(QKᵀ)V with φ(Q)(φ(K)ᵀV) so attention is an associative sum computable in O(n) time with a constant-size recurrent state Sₜ = Σ… |
| [Linear-Time-Invariant SSM (S4 / HiPPO / DPLR)](linear-time-invariant-ssm.md) | A continuous linear ODE x'(t)=Ax+Bu, y=Cx discretized into a fixed-kernel sequence model whose A is initialized by HiPPO (optimal polynomial memory)… |
| [Selective State-Space Models / Mamba](selective-state-space-models-mamba.md) | SSMs whose (B, C, Δ) are made functions of the input — content-based gating lets the recurrence selectively remember or forget tokens, closing the… |
| [State-Space Duality (SSD / Mamba-2)](state-space-duality.md) | Framework proving selective SSMs and a causal-masked form of linear attention are two views of the same structured (semiseparable) matrix transform,… |
| [Recurrence-Convolution-Scan Duality](recurrence-convolution-scan-duality.md) | A linear recurrence h_t = A h_{t-1} + B x_t can be unrolled into a convolution (a fixed kernel over the whole sequence) or computed by an associative… |
| [Hardware-Aware Parallel Scan](hardware-aware-parallel-scan.md) | A work-efficient associative (Blelloch) prefix scan fused into a single CUDA kernel that keeps SSM recurrent states in SRAM/registers — turning… |
| [Chunked / Chunkwise-Parallel Form](chunked-chunkwise-parallel-form.md) | Split the sequence into chunks; compute exact intra-chunk attention quadratically and carry a summarized recurrent state across chunks linearly —… |
| [Gated Linear Attention & Data-Dependent Decay](gated-linear-attention-data-dependent-decay.md) | Replace softmax attention's exact recall with a recurrent matrix-valued memory state S that is multiplicatively decayed by a (often input-dependent)… |
| [Delta Rule / Fast-Weight Update (DeltaNet)](delta-rule-fast-weight-update.md) | Replace linear attention's pure-additive state write (S ← S + vkᵀ) with an error-correcting delta rule S ← S(I − βkkᵀ) + βvkᵀ, so the recurrent… |
| [Test-Time Training / Expressive-State Memory (TTT, Titans)](test-time-training-expressive-state-memory.md) | Reframe the RNN hidden state as a small learnable model (matrix or MLP) whose weights ARE the memory, updated by online (self-supervised) gradient… |
| [RWKV (Time-Mixing Recurrence)](rwkv.md) | An attention-free architecture (RWKV-4→7) that interleaves a token-shift channel-mix with a WKV time-mix behaving like decaying linear attention —… |
| [Long-Convolution Models (Hyena / H3)](long-convolution-models.md) | Replace attention with data-controlled gating of an implicitly-parameterized, sub-quadratic global convolution whose filter spans the whole sequence… |
| [Associative Recall & the Recall-State-Size Tradeoff](associative-recall-the-recall-state-size.md) | MQAR (multi-query associative recall) is the diagnostic for whether a sub-quadratic model can store arbitrary in-context key→value bindings; a… |
| [Hybrid Attention-SSM Architectures](hybrid-attention-ssm-architectures.md) | Interleave a few full-attention layers among many SSM / linear-attention layers (Jamba, Zamba, Samba, Nemotron-H) so the rare attention layers… |
| [Transformer-to-Recurrent Distillation / Linearization](transformer-to-recurrent-distillation.md) | Convert a pretrained softmax-attention transformer into a linear-attention/SSM student by reusing the teacher's weights and matching it… |
| [Constant-Memory Autoregressive Inference (No KV Cache)](constant-memory-autoregressive-inference.md) | The headline serving advantage of recurrent/SSM models — a fixed-size hidden state replaces the linearly-growing transformer KV cache, giving O(1)… |
| [Bidirectional / Multi-Directional SSM Scanning](bidirectional-multi-directional-ssm-scanning.md) | Adapt the inherently-causal selective-SSM (Mamba) recurrence to non-sequential data by running multiple directed scans over a flattened token grid… |

### Pretraining & Data
| Brick | One-liner |
|-------|-----------|
| [Tokenizer Design (BPE / Byte-Level / Tokenizer-Free)](tokenizer-design.md) | The choice of subword algorithm (BPE, byte-level BPE, Unigram-LM) and vocab size V fixes compression (bytes/token), fertility (tokens/word), and the… |
| [Near-Duplicate Deduplication (MinHash / SemDeDup)](near-duplicate-deduplication.md) | Remove exact, fuzzy, and semantic duplicates from pretraining corpora via suffix-array substring matching, MinHash-LSH, and embedding-cluster pruning… |
| [Model-Based Quality Filtering](model-based-quality-filtering.md) | Train a lightweight classifier to score web documents and keep only high-quality ones (FineWeb-Edu's educational-value regressor, DCLM's fastText… |
| [Data Mixture / Domain Weighting (DoReMi, RegMix)](data-mixture-domain-weighting.md) | Set the pretraining sampling proportions across domains (web, code, math, books, multilingual) via proxy-model methods — Group DRO (DoReMi), gradient… |
| [Data Mixing Laws](data-mixing-laws.md) | Predictive scaling laws expressing validation loss as a function of domain mixture proportions (plus N and D), fit on cheap small runs to extrapolate… |
| [Data-Constrained Scaling & Repetition Laws](data-constrained-scaling-repetition-laws.md) | Extension of Chinchilla to repeated data — each extra epoch on the same tokens has exponentially decaying value, with up to ~4 epochs nearly as good… |
| [Synthetic Data & Web Rephrasing (WRAP, Phi)](synthetic-data-web-rephrasing.md) | Use an LLM to rewrite raw web text into clean styles (WRAP) or generate textbook-quality/math/code corpora from scratch (Phi), trading inference… |
| [Model Collapse / Curse of Recursion](model-collapse-curse-of-recursion.md) | Training generation n+1 on the outputs of generation n drives a degenerative feedback loop — tails are forgotten first, variance shrinks, then the… |
| [Annealing / Mid-Training High-Quality Phase](annealing-mid-training-high-quality-phase.md) | A short final pretraining phase (~last few % of tokens) that upweights the highest-quality and benchmark-relevant data (code, math, instructions,… |
| [Pretraining Data Curriculum](pretraining-data-curriculum.md) | Deliberately ordering or phasing the training data — quality/difficulty ramps, domain phase-ins, source up-weighting — instead of i.i.d. sampling… |
| [Decontamination (Train-Test Overlap Removal)](decontamination.md) | Detect and strip pretraining documents that overlap eval benchmarks (n-gram / substring / embedding / canary matching) so reported scores reflect… |
| [Sequence Packing & Document Attention Masking](sequence-packing-document-attention-masking.md) | Concatenate variable-length documents into fixed-length training sequences to eliminate padding waste, with a block-diagonal attention mask (and… |
| [Teacher Forcing & Exposure Bias](teacher-forcing-exposure-bias.md) | Teacher forcing trains autoregressive models on ground-truth prefixes (each step conditioned on real previous tokens, not the model's own); exposure… |

### Scaling Laws & Compute Allocation
| Brick | One-liner |
|-------|-----------|
| [Scaling Laws](scaling-laws.md) | Loss follows power laws in model size (N), dataset size (D), and compute (C = 6ND); Chinchilla says train a smaller model on more data; 2024+ labs… |
| [Compute Budget Identity C ≈ 6ND](compute-budget-identity-c-6nd.md) | Total training FLOPs are roughly six times non-embedding parameters times training tokens — the back-of-envelope behind every token budget, GPU-hour… |
| [Inference-Optimal Over-Training](inference-optimal-over-training.md) | Deliberately train a *smaller* model on far *more* tokens than Chinchilla-optimal (often 100-1000:1 token-to-param ratios) because lifetime inference… |
| [Emergent Abilities & the Mirage Critique](emergent-abilities-the-mirage-critique.md) | Sharp, seemingly-discontinuous capability jumps with scale — argued by Schaeffer et al. (2023) to be largely artifacts of nonlinear/discontinuous… |
| [Capacity-Density & Capability Scaling](capacity-density-capability-scaling.md) | Beyond loss-vs-scale, newer fits relate *downstream capability* to compute, and "capacity density" measures effective capability per parameter —… |
| [Scaling-Law Extrapolation & Brittleness](scaling-law-extrapolation-brittleness.md) | Fitting a power law on small/cheap runs to predict the loss of a much larger run — reliable only when the fitted exponent, irreducible-loss offset,… |

### Optimization & Training Dynamics
| Brick | One-liner |
|-------|-----------|
| [AdamW](adamw.md) | Adam with *decoupled* weight decay — the L2 penalty is applied directly to the weights as a separate shrinkage step instead of being added into the… |
| [Muon Optimizer](muon-optimizer.md) | Momentum SGD for 2D weight matrices whose update is orthogonalized via a few Newton-Schulz iterations (cheap approximate `U V^T` from the gradient's… |
| [Shampoo / SOAP (Second-Order Preconditioners)](shampoo-soap.md) | Adaptive optimizers that precondition each weight matrix with Kronecker-factored second-moment statistics instead of Adam's per-element scaling; SOAP… |
| [Adam Hyperparameters (betas, epsilon, bias correction)](adam-hyperparameters.md) | β1 (momentum decay), β2 (second-moment decay), and ε (denominator floor) set Adam's per-coordinate adaptive step, with the 1−βᵗ bias correction… |
| [maximal update parameterization (muP)](maximal-update-parameterization.md) | Width-aware scaling of init variances and per-layer learning rates (Tensor Programs framework) that holds the network in the feature-learning regime… |
| [Warmup-Stable-Decay (WSD) Schedule](warmup-stable-decay-schedule.md) | Three-phase LR schedule — linear warmup, a long constant plateau at peak LR, then a short fast decay — that decouples the schedule from a fixed total… |
| [Learning-Rate Warmup](learning-rate-warmup.md) | Linearly ramp the LR from ~0 to its peak over the first W steps to avoid early divergence from large, poorly-conditioned updates and to let Adam's… |
| [Cosine LR Decay](cosine-lr-decay.md) | Anneal the learning rate along a half-cosine curve from peak to a small floor over a pre-committed token/step horizon — the Chinchilla-era… |
| [Decoupled Weight Decay](decoupled-weight-decay.md) | Multiplicative per-step shrinkage of weights applied directly to the parameter, decoupled from the loss gradient; under Adam it behaves less like… |
| [Gradient Clipping](gradient-clipping.md) | Rescale the gradient — almost always by its global L2 norm — down to a threshold c so no single update can blow up, capping the damage from rare… |
| [Loss Spikes & Training Instability](loss-spikes-training-instability.md) | Sudden mid-training loss divergences at scale, traced to attention-logit blowup, exploding output/activation norms, bf16 precision limits, or bad… |
| [Z-Loss & Logit Stabilization](z-loss-logit-stabilization.md) | A family of auxiliary regularizers — output/router z-loss penalizing the softmax log-partition, attention-logit soft-capping, and embedding/logit… |
| [Edge of Stability](edge-of-stability.md) | During full-batch gradient descent, the top Hessian eigenvalue (sharpness) climbs to ≈2/η and then hovers just above it — training stably reduces… |
| [Sharpness-Aware Minimization (SAM) & Flat Minima](sharpness-aware-minimization-flat-minima.md) | Flat minima (low loss curvature in a neighborhood) tend to generalize better; SAM optimizes the worst-case loss in an ε-ball around the weights via a… |
| [Critical Batch Size & Gradient Noise Scale](critical-batch-size-gradient-noise-scale.md) | The largest batch that still yields near-linear speedup in steps-to-target-loss, predictable from the gradient noise scale (signal-to-noise ratio of… |
| [LR-Batch-Size Coupling (Linear / Sqrt Scaling)](lr-batch-size-coupling.md) | To preserve optimization dynamics as you grow the (data-parallel) batch size B, scale the learning rate with B — linearly (η ∝ B) for SGD/momentum,… |
| [Initialization & Residual/Depth Scaling](initialization-residual-depth-scaling.md) | Choose weight-init variances and downscale residual branches (typically by 1/√(2L) or per-layer) so that at step zero forward activation variance and… |
| [Effective LR & Norm-Growth Dynamics](effective-lr-norm-growth-dynamics.md) | Under scale-invariant normalization (LN/RMSNorm) the loss depends only on weight *direction*, so the meaningful step size is the effective LR η_eff ≈… |
| [Low-Precision Optimizer States & Stochastic Rounding](low-precision-optimizer-states-stochastic.md) | Store optimizer moments (and sometimes master weights) in 8-bit/fp8 with per-block dynamic quantization, and apply stochastic rounding to bf16/fp16… |
| [Hyperparameter Scaling Laws (LR / batch / wd)](hyperparameter-scaling-laws.md) | Fitted power laws giving optimal learning rate, batch size, and (independent) weight decay as functions of model size N and tokens D, so HPs are… |
| [Implicit Bias of SGD](implicit-bias.md) | Among the infinitely many parameter settings that perfectly fit the data, GD/SGD doesn't pick one at random — it converges to structured,… |
| [Grokking](grokking.md) | A delayed generalization phenomenon where test accuracy jumps from chance to near-perfect long after training accuracy has saturated at 100% — a… |
| [Adam Update Rule (the m/v EMA equations)](adam-update-rule.md) | Per-parameter adaptive optimizer that keeps EMAs of the gradient (1st moment m) and its square (2nd moment v), bias-corrects both, and steps θ ← θ −… |
| [Dropout](dropout.md) | Stochastic regularization that randomly zeros activations during training (and rescales the survivors) so the network can't co-adapt units — still… |
| [Label Smoothing](label-smoothing.md) | Replace one-hot targets with a mixture of the hard label and a uniform distribution — softening the target curbs overconfidence, improves… |
| [Weight Averaging / EMA & Model Soups](weight-averaging-ema-model-soups.md) | Average weights — either across training steps (EMA) or across independently fine-tuned checkpoints (soups) — to land in a flatter,… |
| [Muon / Spectral-Norm & Newton-Schulz Orthogonalization](muon-spectral-norm-newton-schulz.md) | Replace a momentum-matrix gradient G with its nearest semi-orthogonal matrix UVᵀ (all singular values pushed to 1), computed cheaply by a few… |
| [Fisher Information & Natural Gradient](fisher-information-natural-gradient.md) | The Fisher F is the curvature of the log-likelihood (expected outer product of score, = expected Hessian of NLL); preconditioning the gradient by F⁻¹… |

### Learning Theory & Generalization
| Brick | One-liner |
|-------|-----------|
| [Bias–Variance Tradeoff](bias-variance.md) | Expected test error decomposes as Bias² + Variance + irreducible Noise; the classical claim that capacity trades bias for variance (the U-shape)… |
| [Double Descent](double-descent.md) | Test error falls, then rises to a peak exactly at the interpolation threshold (params ≈ training points), then falls again as the model grows further… |
| [Neural Tangent Kernel (NTK)](ntk.md) | In the infinite-width limit with the right parametrization, the network stays near its initialization and training becomes kernel regression with a… |
| [Feature Learning vs Lazy Training](feature-learning-vs-lazy-training.md) | A dichotomy in training dynamics — whether internal representations actually move (rich / feature-learning regime) or stay frozen at init while only… |
| [PAC-Bayes](pac-bayes.md) | Generalization bounds on the *expected* risk of a stochastic (Gibbs) classifier, controlled by KL(Q‖P) between a learned posterior Q and a… |
| [Information Bottleneck](information-bottleneck.md) | Learn a representation T that maximally compresses X (min I(X;T)) while retaining everything predictive of Y (max I(T;Y)); the trade-off curve is the… |
| [Memorization vs Generalization](memorization-vs-generalization.md) | LLMs provably store verbatim training data — extractable via prompting, scaling super-linearly with duplication and model size — a distinct… |
| [In-Context Learning](in-context-learning.md) | LLMs solve new tasks from prompt examples (x₁,y₁,…,x_k,y_k, x_query) without any weight update — interpretable as implicit Bayesian inference over a… |

### Post-Training & RL Alignment
| Brick | One-liner |
|-------|-----------|
| [SFT / Instruction Tuning](sft-instruction-tuning.md) | Supervised fine-tuning on (instruction, response) demonstrations via next-token cross-entropy to teach format and behavior; stage zero of every… |
| [RLHF](rlhf.md) | Align a model to human preferences via three stages — SFT, then a reward model trained on pairwise comparisons, then PPO that maximizes reward minus… |
| [Reward Modeling (Bradley-Terry)](reward-modeling.md) | Learn a scalar reward r_φ(x,y) from pairwise human preferences via a Bradley-Terry logistic loss so the chosen response scores above the rejected one… |
| [PPO Clipped Surrogate Objective](ppo-clipped-surrogate-objective.md) | Constrain each policy-gradient update with a clipped importance ratio — max the min of unclipped and clipped advantage terms — to stay… |
| [Direct Preference Optimization (DPO)](dpo.md) | Closed-form preference training — fit the policy directly on preference pairs with a binary-classification loss whose implicit reward is the policy's… |
| [DPO Variants (IPO / KTO / ORPO / SimPO)](dpo-variants.md) | The post-DPO zoo, each patching one DPO weakness — IPO replaces the logistic loss with a squared margin to stop preference-overfitting, KTO trains on… |
| [GRPO — Group Relative Policy Optimization](grpo.md) | Critic-free RL for LLMs — estimate the advantage baseline from a group of sampled responses instead of a value network; combined with verifiable… |
| [RLVR (RL with Verifiable Rewards)](rlvr.md) | Replace the learned reward model with a programmatic verifier (math answer-checking, unit tests, format/regex checks) returning a… |
| [KL Regularization to Reference Policy](kl-regularization-to-reference-policy.md) | Penalize divergence from the frozen SFT/reference model (β·KL[π_θ‖π_ref]) during RL so the policy chases reward without collapsing into degenerate,… |
| [Reward Hacking / Over-Optimization](reward-hacking-over-optimization.md) | A Goodhart-law failure where the policy exploits flaws in an imperfect proxy reward (length bias, sycophancy, formatting tells, verifier loopholes),… |
| [RLAIF / Constitutional AI](rlaif-constitutional-ai.md) | Replace human preference labels with AI feedback — a model self-critiques and revises its outputs against a written "constitution" of principles,… |
| [Rejection Sampling / Best-of-N](rejection-sampling-best-of-n.md) | Sample N candidates from a policy, score each with a reward model or verifier, keep the top one(s); used at inference (BoN) or to filter SFT data for… |
| [Length Normalization / Bias Control](length-normalization-bias-control.md) | Counteract the systematic bias of preference and RL objectives toward longer outputs — because sequence log-probs and reward models both correlate… |
| [Process vs Outcome Reward Models (PRM / ORM)](process-vs-outcome-reward-models.md) | ORMs score only the final answer; PRMs score each intermediate reasoning step for denser credit assignment — the core supervision-design axis in… |
| [Preference Data & Annotation Pipeline](preference-data-annotation-pipeline.md) | The end-to-end construction of comparison datasets — prompt sourcing, response sampling, pairwise/k-wise labeling, on- vs off-policy generation, and… |
| [Iterative / Online DPO](iterative-online-dpo.md) | Close DPO's distribution-mismatch gap by looping {sample on-policy from the current model → label fresh pairs with a judge/RM → run a round of DPO},… |
| [Verifier Design & Reward Shaping (RLVR)](verifier-design-reward-shaping.md) | In verifiable-reward RL the checker *is* the alignment target, so engineering it — answer extraction/canonicalization, sandboxed unit-test harnesses,… |
| [Entropy Collapse & Exploration Control](entropy-collapse-exploration-control.md) | During RLVR/RLHF the policy's token-level output entropy can collapse toward a deterministic mode, killing exploration and locking in pass@1 at the… |
| [Gradient Variance Reduction & Control Variates](gradient-variance-reduction-control-variates.md) | Subtract a correlated, zero-mean term (baseline / control variate) from a Monte Carlo gradient estimator to slash its variance without adding bias —… |
| [Generalized Advantage Estimation (GAE)](generalized-advantage-estimation.md) | λ-weighted exponential average of multi-step TD residuals that estimates the advantage A(s,a) with a tunable bias-variance knob — the "A" that PPO's… |
| [Importance Sampling & the Off-Policy Ratio](importance-sampling-the-off-policy-ratio.md) | Reweight samples drawn from a behavior distribution q to estimate an expectation under a target distribution p via the ratio w = p/q; in RL… |
| [Catastrophic Forgetting & Continual Learning](catastrophic-forgetting-continual-learning.md) | When you train on task B, gradient descent overwrites the weights encoding task A — the central tension in post-training, model editing, and… |
| [Grouped / Document-Aware Loss Masking & Token Weighting](grouped-document-aware-loss-masking-token.md) | A per-token mask/weight vector that decides which tokens contribute to the next-token loss and how much — completion-only SFT masks the prompt,… |
| [Speculative / Asynchronous RL Rollout Infrastructure](speculative-asynchronous-rl-rollout.md) | Decouple generation (rollout workers) from policy optimization (trainer) so GPUs never idle; correct for the resulting off-policy staleness with… |

### Reasoning & Test-Time Compute
| Brick | One-liner |
|-------|-----------|
| [Chain-of-Thought & Test-Time Compute](chain-of-thought.md) | Elicit intermediate reasoning tokens before the answer, then spend more inference compute (longer chains, sampling, search, verification) to trade… |
| [Self-Consistency](self-consistency.md) | Sample K chain-of-thought traces at nonzero temperature, then majority-vote over the *final answers* (marginalizing out the reasoning paths) instead… |
| [Test-Time Compute Scaling](test-time-compute-scaling.md) | Inference compute — more samples, longer chains, or search — is a separate scaling axis that predictably lowers error, and per the compute-optimal… |
| [Long Reasoning Chains (o1 / R1-style)](long-reasoning-chains.md) | Models RL-trained (on verifiable rewards) to emit very long internal CoT — exploring, backtracking, and self-verifying before answering — so accuracy… |
| [Generator-Verifier Gap](generator-verifier-gap.md) | The asymmetry that checking a candidate solution is cheaper/more reliable than producing it — the conceptual reason test-time compute (best-of-N,… |
| [Tree-of-Thought (ToT)](tree-of-thought.md) | Reframe reasoning as deliberate search over a tree of partial "thought" states — generate candidate next-thoughts, score them with a value heuristic,… |
| [MCTS-Style Inference Search (LATS)](mcts-style-inference-search.md) | Run Monte Carlo Tree Search over reasoning/action steps — selection → expansion → simulation → backup — using an LLM as the policy and a value… |
| [Self-Correction / Reflection](self-correction-reflection.md) | A model critiques and revises its own output across one or more passes; reliably helpful when the critique is grounded in external/ground-truth… |
| [Self-Improvement / STaR Bootstrapping](self-improvement-star-bootstrapping.md) | Iteratively sample reasoning traces from the current model, keep only those that reach the verified-correct answer (optionally rationalize failures… |
| [Reasoning Distillation](reasoning-distillation.md) | Teach a small model to reason by SFT on long chain-of-thought traces sampled from a strong reasoner (e.g., DeepSeek-R1), trading expensive RLVR for… |
| [Compute-Optimal Test-Time Allocation](compute-optimal-test-time-allocation.md) | Given a fixed inference budget, pick the test-time strategy (more parallel samples vs. deeper sequential revision vs. tree search) and how hard to… |
| [Budget Forcing / Thinking-Token Control](budget-forcing-thinking-token-control.md) | A test-time decoding intervention that explicitly sets the reasoning-token budget — either capping the thinking trace by force-injecting an… |
| [CoT Faithfulness / Monitorability](cot-faithfulness-monitorability.md) | Whether a model's verbalized chain-of-thought actually reflects the computation that produced its answer — determining if reading the CoT is a valid… |
| [Latent / Continuous Reasoning](latent-continuous-reasoning.md) | Reasoning in continuous latent space — feeding the model's last hidden state back as the next input embedding (Coconut) or adding recurrent depth at… |

### Inference & Serving
| Brick | One-liner |
|-------|-----------|
| [KV Cache](kv-cache.md) | Cache the K,V projections of all past tokens so each autoregressive decode step is O(1) in sequence length instead of O(N); the cache grows linearly… |
| [Prefill vs Decode](prefill-vs-decode.md) | The two inference phases — prefill ingests the whole prompt in one compute-bound parallel pass (sets TTFT), decode emits one token per step and is… |
| [PagedAttention](pagedattention.md) | Store the KV cache in fixed-size non-contiguous physical blocks indexed by a per-sequence block table (OS-style virtual paging), eliminating… |
| [Continuous Batching](continuous-batching.md) | Iteration-level (token-step) scheduling that admits new requests and retires finished ones at every decode step instead of running a static batch to… |
| [Prefix Caching / RadixAttention](prefix-caching-radixattention.md) | Reuse already-computed KV-cache blocks for shared prompt prefixes (system prompts, few-shot exemplars, conversation history, RAG docs) keyed by a… |
| [Chunked Prefill](chunked-prefill.md) | Split a long prompt's prefill into fixed-size token chunks and interleave them with ongoing decode steps in the same batch, capping the latency spike… |
| [Disaggregated Prefill/Decode](disaggregated-prefill-decode.md) | Run prefill and decode on separate GPU pools — each with its own parallelism, batch policy, and replica count — and ship the KV cache between them,… |
| [Speculative Decoding](speculative-decoding.md) | A cheap draft model proposes k tokens autoregressively; the expensive target model verifies all k in one parallel forward pass and accepts a prefix… |
| [EAGLE / Medusa Self-Speculation](eagle-medusa-self-speculation.md) | Speculative decoding without a separate draft model — bolt lightweight prediction heads onto the target model itself (Medusa) or a tiny… |
| [Tree / Token-Tree Verification](tree-token-tree-verification.md) | Pack many candidate continuations as a token tree and verify them all in a single draft-model-free target forward pass via a custom (tree) attention… |
| [Latency-Throughput Tradeoff (TTFT / TPOT / Goodput / SLO)](latency-throughput-tradeoff.md) | LLM serving lives on a Pareto frontier trading per-request latency (TTFT, inter-token latency) against aggregate token throughput; the metric the… |
| [KV-Cache Quantization](kv-cache-quantization.md) | Store cached K/V in low precision (FP8/INT8/INT4) with the right granularity — per-channel for keys, per-token for values — to halve or quarter KV… |
| [KV-Cache Compression / Eviction](kv-cache-compression-eviction.md) | Bound KV-cache memory by keeping only a subset of token K,V (attention sinks + recent window, or top-scoring tokens) and dropping/merging the rest,… |
| [Hierarchical / Distributed KV Store (Offload & Transfer)](hierarchical-distributed-kv-store.md) | Treat the KV cache as a tiered, transferable resource — spill cold blocks down a GPU HBM → host DRAM → NVMe hierarchy and move/share them across… |
| [Inference-Time Parallelism (TP / PP / EP for Serving)](inference-time-parallelism.md) | Split weights/experts across GPUs at inference: tensor parallelism (TP) shards every matmul to cut latency, pipeline parallelism (PP) stages layers… |
| [Continuous-Batch Scheduler & Admission Control](continuous-batch-scheduler-admission-control.md) | The per-iteration serving control loop that picks which running/waiting requests to prefill vs decode, how to chunk and pack them into a token… |
| [FP8 / Low-Bit Inference & Fused Kernels](fp8-low-bit-inference-fused-kernels.md) | Serve weights/activations (and increasingly attention) in FP8 or sub-byte INT with calibrated per-tensor/per-channel scales, fused into dequant-GEMM… |
| [Decoding / Sampling Strategies (Greedy, Beam, Top-k, Top-p, Min-p)](decoding-sampling-strategies.md) | How a token is actually picked from the next-token distribution at inference — deterministic (greedy/beam) vs. truncated stochastic (top-k,… |
| [CUDA Graphs & Kernel-Launch Overhead](cuda-graphs-kernel-launch-overhead.md) | Capture a fixed sequence of GPU kernels once into a replayable graph, then relaunch the whole graph with a single API call — eliminating the… |

### Distributed Training & Systems
| Brick | One-liner |
|-------|-----------|
| [Data Parallelism (DDP)](data-parallelism.md) | Replicate the full model on every one of N GPUs, split each global batch into N per-GPU shards, and `all_reduce` the gradients every step so all… |
| [ZeRO](zero.md) | Zero Redundancy Optimizer — shard the optimizer state (stage 1), gradients (stage 2), and parameters (stage 3) across data-parallel ranks instead of… |
| [FSDP — Fully Sharded Data Parallel](fsdp.md) | PyTorch's native ZeRO-3 — shards params, grads, and optimizer state across N ranks, all-gathers each layer's full weights just-in-time for compute,… |
| [Tensor Parallelism](tensor-parallel.md) | Intra-layer model parallelism (Megatron-LM): shard individual weight matrices across GPUs so each device computes a slice of every layer, fused with… |
| [Pipeline Parallelism & the Bubble](pipeline-parallelism-the-bubble.md) | Partition layers into stages across GPUs and stream micro-batches through them; the schedule (GPipe, 1F1B, interleaved, zero-bubble/DualPipe) trades… |
| [Sequence / Context Parallelism](sequence-context-parallelism.md) | Shard the sequence (token) dimension across GPUs so each device holds only part of the context; Ring Attention streams K/V around the ring and… |
| [Expert Parallelism (MoE)](expert-parallelism.md) | Shard MoE experts across GPUs and route tokens to them via all-to-all dispatch/combine collectives, scaling total parameters without scaling… |
| [3D / nD Parallelism](3d-nd-parallelism.md) | Compose data + tensor + pipeline (+ sequence + expert) parallelism along orthogonal axes of a logical device mesh so a trillion-parameter model fits… |
| [Collective Communication Primitives](collective-communication-primitives.md) | The fixed set of GPU group-communication operations (all-reduce, reduce-scatter, all-gather, all-to-all, broadcast) whose ring/tree algorithms set… |
| [Computation-Communication Overlap](computation-communication-overlap.md) | Hide collective latency by issuing comms on separate CUDA streams that run concurrently with compute (prefetch all-gather, bucket gradient… |
| [GPU Memory Hierarchy (HBM / SRAM / Registers)](gpu-memory-hierarchy.md) | A tiered memory stack — registers, L1/shared SRAM, L2, then HBM — where each level up is ~10-100x faster but ~100-1000x smaller, so data movement… |
| [Roofline Model](roofline.md) | Every GPU operation is either compute-bound or memory-bandwidth-bound; the roofline tells you which, and therefore how to optimize it. |
| [Kernel Fusion](kernel-fusion.md) | Merge multiple elementwise/reduction ops into a single GPU kernel so intermediates stay in registers/SRAM instead of round-tripping through HBM — the… |
| [FlashAttention](flash-attention.md) | IO-aware exact attention via tiling — computes the same result as standard attention but reads/writes HBM O(N) times instead of O(N²), giving 2-4×… |
| [Triton](triton.md) | A Python-embedded DSL for writing GPU kernels at the *block* level — you express what one program instance does to a tile of data; the compiler… |
| [Tensor / Tile-MMA Cores](tensor-tile-mma-cores.md) | Specialized matrix-multiply-accumulate units (NVIDIA Tensor Cores) that deliver the bulk of a GPU's FLOP/s for GEMMs; on Hopper/Blackwell they are… |
| [GPU Interconnect Topology (NVLink / NVSwitch / InfiniBand)](gpu-interconnect-topology.md) | A bandwidth hierarchy — NVLink/NVSwitch give ~TB/s all-to-all *within* a node while InfiniBand/Ethernet give ~tens-of-GB/s *across* nodes — so… |
| [MFU / Model FLOPs Utilization](mfu-model-flops-utilization.md) | Achieved model-FLOPs throughput as a fraction of hardware peak — MFU = (6ND_throughput) / peak_FLOP/s — the universal scorecard every systems and… |
| [Gradient Accumulation & Micro-Batching](gradient-accumulation-micro-batching.md) | Split a large effective batch into K micro-batches, run forward/backward on each and sum (or average) the gradients before a single optimizer step —… |
| [Online Softmax / Safe-Softmax Recurrence](online-softmax-safe-softmax-recurrence.md) | A single-pass, numerically-stable recurrence that maintains a running max and running denominator so softmax (and its weighted output) can be… |
| [Ring Attention / Blockwise Distributed Attention](ring-attention-blockwise-distributed-attention.md) | Shard the sequence across devices and pass KV blocks hop-by-hop around a ring while each device computes its local attention block — exact attention… |

### Efficiency & Compression
| Brick | One-liner |
|-------|-----------|
| [LoRA — Low-Rank Adaptation](lora.md) | Fine-tune only a low-rank decomposition ΔW = AB (r << min(d,k)); freeze the pretrained weights; dramatically reduces trainable parameters. |
| [QLoRA](qlora.md) | LoRA adapters trained on top of a frozen 4-bit NF4-quantized base, with double quantization and paged optimizers — fine-tunes a 65B model on one 48GB… |
| [DoRA (Weight-Decomposed LoRA)](dora.md) | Split each pretrained weight into magnitude (a per-column norm) and direction (the unit vector), train the magnitude directly while adapting the… |
| [Adapters & Prefix/Prompt Tuning](adapters-prefix-prompt-tuning.md) | The pre-LoRA PEFT family — freeze base weights and train either small bottleneck modules inserted between layers (adapters) or learned virtual tokens… |
| [LoRA Merging & Multi-Adapter Serving](lora-merging-multi-adapter-serving.md) | Either fold ΔW=(α/r)BA back into W₀ for zero-overhead single-tenant inference, or keep adapters unmerged and hot-swap/batch hundreds of them off one… |
| [Quantization](quantization.md) | Store/compute weights and activations in low-bit formats (INT8/INT4/FP8) to cut memory and boost throughput; PTQ (cheap, post-hoc) vs QAT (trains… |
| [GPTQ](gptq.md) | One-shot post-training weight quantization to 3-4 bits that quantizes each layer's weights column-by-column while using approximate second-order… |
| [AWQ (Activation-Aware Weight Quantization)](awq.md) | Weight-only PTQ that protects the salient ~1% of weight channels — identified by *activation* magnitude, not weight magnitude — via a per-channel… |
| [Activation Outliers / SmoothQuant](activation-outliers-smoothquant.md) | A handful of feature channels in LLM activations grow to 10-100x the typical magnitude (concentrated, persistent, emerging at ~6.7B params), wrecking… |
| [Quantization-Aware Training (QAT)](quantization-aware-training.md) | Insert fake-quant ops in the forward pass and backprop through them with the straight-through estimator, so weights learn to live on the low-bit grid… |
| [Extreme / Sub-4-Bit Quantization (2-bit, Ternary, BitNet)](extreme-sub-4-bit-quantization.md) | Push weights below 4 bits — 2-bit, ternary {−1,0,1} (BitNet b1.58 ≈ 1.58 bits), or binary — via learned scales, incoherence/rotation processing… |
| [Mixed-Precision Bit Allocation](mixed-precision-bit-allocation.md) | Assign different bitwidths per layer/channel/group by a sensitivity score (Hessian-trace or gradient-based), keeping the few sensitive components at… |
| [FP8 / Low-Precision Training](fp8-low-precision-training.md) | Train with 8-bit floats — E4M3 for forward/weights/activations, E5M2 for gradients — using per-tensor or fine-grained microscaling factors to keep… |
| [Mixed Precision](mixed-precision.md) | Run compute in 16-bit (FP16/BF16) for speed and memory while keeping an FP32 master copy of weights — and for FP16, loss scaling — to preserve… |
| [Gradient Checkpointing](gradient-checkpointing.md) | Recompute activations during the backward pass instead of storing them — trade ~1 extra forward pass for an O(√L) (or better) reduction in activation… |
| [Knowledge Distillation](knowledge-distillation.md) | Train a small student to match a large teacher's soft output distribution (and often intermediate features/attention), transferring "dark knowledge"… |
| [Structured vs Unstructured Pruning](structured-vs-unstructured-pruning.md) | Remove weights in hardware-friendly patterns (heads, channels, layers, 2:4 blocks) for real wall-clock speedup, or arbitrarily by importance… |
| [2:4 Semi-Structured Sparsity](2-4-semi-structured-sparsity.md) | Force exactly 2 of every 4 contiguous weights to zero so NVIDIA's Sparse Tensor Cores skip the zeros and deliver ~2x matmul throughput — the only… |
| [SparseGPT / Wanda One-Shot Pruning](sparsegpt-wanda-one-shot-pruning.md) | Prune LLMs to 50%+ sparsity in one shot, no retraining — SparseGPT solves a layer-wise Hessian-based reconstruction per weight, Wanda uses a… |
| [Task Arithmetic / Task Vectors](task-arithmetic-task-vectors.md) | A task vector is the weight delta τ = θ_ft − θ_base from fine-tuning; behaviors compose linearly in weight space (add to acquire, negate to forget),… |
| [BF16 vs FP16 vs FP32 (Floating-Point Formats)](bf16-vs-fp16-vs-fp32.md) | Three IEEE-ish float layouts trading mantissa bits (precision) against exponent bits (dynamic range); bf16 keeps fp32's 8-bit exponent so it never… |
| [MXFP / Microscaling Block Formats (FP4/FP6/FP8)](mxfp-microscaling-block-formats.md) | OCP-standardized low-precision formats where a small block of K elements shares one scale factor, so each value stores only a tiny FP… |
| [Hadamard / Rotation-Based Quantization (QuaRot, SpinQuant)](hadamard-rotation-based-quantization.md) | Multiply weights and activations by orthogonal (often Hadamard) matrices that are computationally inert (Q Qᵀ = I) but spread energy across channels,… |

### Generative Models (Diffusion / Flow)
| Brick | One-liner |
|-------|-----------|
| [DDPM (Denoising Diffusion)](ddpm.md) | A forward process gradually corrupts data into Gaussian noise over T steps; a network is trained to predict that noise, and sampling reverses the… |
| [Score-Based Models / SDEs](score-based-models-sdes.md) | Generation as learning the score ∇ₓ log p(x) and integrating a reverse-time SDE (or the equivalent probability-flow ODE); the continuous-time… |
| [Classifier-Free Guidance (CFG)](classifier-free-guidance.md) | Jointly train conditional and unconditional models via random condition-dropout, then at sampling extrapolate the conditional prediction away from… |
| [Latent Diffusion (LDM)](latent-diffusion.md) | Run the diffusion process in the compressed latent space z = E(x) of a frozen, pretrained autoencoder instead of in pixel space — cutting spatial… |
| [Flow Matching](flow-matching.md) | Simulation-free training of continuous normalizing flows by regressing a velocity field that transports noise to data along a prescribed probability… |
| [Consistency Models](consistency-models.md) | A model f_theta(x_t, t) trained to map any point on a probability-flow ODE trajectory directly to that trajectory's origin x_0, enabling 1-step… |
| [Diffusion Distillation (Few-Step)](diffusion-distillation.md) | Compress a slow many-step (50-1000 NFE) diffusion teacher into a 1-8 step student via progressive halving, consistency self-distillation, adversarial… |
| [Diffusion Transformer (DiT)](diffusion-transformer.md) | Replace the U-Net denoiser with a plain transformer over latent patches, conditioned via adaLN-Zero (timestep + class/text inject scale/shift/gate… |
| [Noise Schedules & Timestep Weighting](noise-schedules-timestep-weighting.md) | The β/σ schedule, SNR parameterization, timestep sampling distribution, and resolution-dependent shift jointly decide *which noise levels the model… |
| [EDM Preconditioning & ODE Samplers](edm-preconditioning-ode-samplers.md) | Karras et al. (2022) reframe diffusion as a single noise-level (sigma) continuum with c_skip/c_out/c_in/c_noise network preconditioning, a log-normal… |
| [VAE & the ELBO / Reparameterization Trick](vae-the-elbo-reparameterization-trick.md) | Amortized variational inference that maximizes an evidence lower bound on log p(x), using the reparameterization trick to push low-variance Monte… |
| [GAN Objective & Adversarial Losses](gan-objective-adversarial-losses.md) | A generator-discriminator minimax game (non-saturating / hinge losses, R1 gradient penalty, spectral norm) that learns to draw from p_data without an… |
| [VQ-VAE / Discrete Visual Tokenizers](vq-vae-discrete-visual-tokenizers.md) | Encode an image/video into a grid of discrete codebook indices via nearest-neighbor vector quantization, enabling autoregressive or masked-token… |
| [Masked / Parallel Token Generation (MaskGIT)](masked-parallel-token-generation.md) | Generate a grid of discrete visual tokens non-autoregressively by iteratively unmasking the highest-confidence predictions over a handful of decoding… |
| [Autoregressive Visual Generation (Next-Token / Next-Scale)](autoregressive-visual-generation.md) | Generate images/video/audio as sequences of discrete tokens with a causal transformer (`p(x) = Πᵢ p(xᵢ / x_{<i})`), either token-by-token in raster… |
| [Neural Audio Codecs (RVQ)](neural-audio-codecs.md) | Convolutional autoencoder + residual vector quantization compresses waveform audio into N stacked discrete code streams at ~kbps bitrates… |
| [Discrete / Masked Diffusion Models](discrete-masked-diffusion-models.md) | Diffusion over discrete tokens via an absorbing-state (`[MASK]`) forward process or general continuous-time Markov chains, trained by score-entropy /… |
| [Video Generation: Spatiotemporal Modeling](video-generation-spatiotemporal-modeling.md) | Generate video by diffusing/denoising over spacetime latent patches from a causal 3D tokenizer, modeled with a DiT over factorized or full 3D… |
| [Inference-Time / Reward Guidance for Generators](inference-time-reward-guidance-for-generators.md) | Steer a (usually diffusion/flow) generator toward a reward r(x) — at sampling time via reward-gradient guidance or best-of-N, or by fine-tuning the… |
| [Diffusion Forcing / Block-Autoregressive Diffusion](diffusion-forcing-block-autoregressive-diffusion.md) | Train a sequence model to denoise each token at its own independent noise level, unifying next-token autoregression (full-noise future, clean past)… |

### Interpretability
| Brick | One-liner |
|-------|-----------|
| [Superposition](superposition.md) | Networks represent more features than they have dimensions by encoding them as near-orthogonal directions in activation space, tolerating small… |
| [Features vs Neurons (Polysemanticity)](features-vs-neurons.md) | The model's basis of computation (neurons / residual-stream dimensions) is not its basis of meaning (features); a single neuron is typically… |
| [Linear Representation Hypothesis](linear-representation-hypothesis.md) | The conjecture that high-level concepts are encoded as linear directions in a model's activation (residual stream) space, so features superpose… |
| [Sparse Autoencoders (SAEs)](sparse-autoencoders.md) | Learn an overcomplete, sparsely-activating dictionary over model activations to decompose superposed features into (approximately) monosemantic… |
| [SAE Variants (TopK / JumpReLU / Gated)](sae-variants.md) | Architectural fixes to vanilla L1 sparse autoencoders — TopK enforces exact k-sparsity, JumpReLU learns a per-feature activation threshold, Gated… |
| [SAE Pathologies (Shrinkage, Dead Features, Absorption, Splitting)](sae-pathologies.md) | The four systematic failure modes of sparse-autoencoder training — L1 shrinks active magnitudes, latents die and never fire, broad features get… |
| [Circuits](circuits.md) | A subgraph of model features/components (heads, MLP neurons, SAE features) wired by weights that implements a human-interpretable algorithm — the… |
| [Induction Heads](induction-heads.md) | A two-head circuit — a previous-token head writing token[i−1]'s identity into position i, then an induction head that K-composes on it to attend… |
| [QK / OV Circuits & Head Decomposition](qk-ov-circuits-head-decomposition.md) | Factor an attention head into a query-key circuit W_QK = W_Q W_K^T (which positions attend to which) and an output-value circuit W_OV = W_O W_V (what… |
| [Activation Patching / Causal Tracing](activation-patching-causal-tracing.md) | A causal intervention that copies activations from a clean run into a corrupted run (or vice versa) at specific components, measuring each… |
| [Attribution Patching](attribution-patching.md) | A first-order Taylor approximation of activation patching that estimates the causal effect of swapping every component (or edge) from clean to… |
| [Logit Lens](logit-lens.md) | Decode any intermediate residual-stream activation by projecting it through the final norm + unembedding to read the model's current best-guess token… |
| [Probing Classifiers](probing-classifiers.md) | Train a simple (usually linear) classifier on frozen activations to test whether a property — syntax, truth, sentiment, board state — is linearly… |
| [Steering Vectors / Activation Steering](steering-vectors-activation-steering.md) | Add a fixed direction v to the residual stream at inference — h ← h + αv — to causally push generation toward (or away from) a concept, where v comes… |
| [Transcoders & Attribution Graphs](transcoders-attribution-graphs.md) | A transcoder is a sparse, interpretable module that learns an MLP's input→output map (replacing it for analysis), and stacking these into a… |
| [Automated Interpretability / Auto-Labeling](automated-interpretability-auto-labeling.md) | Use an LLM to generate a natural-language explanation for what a feature/neuron does, then *score* that explanation by how well it predicts the… |
| [Faithfulness & Completeness of Explanations](faithfulness-completeness-of-explanations.md) | Two orthogonal axes for judging an interpretability claim — *faithful* = the explanation reflects the mechanism the model actually uses (causal, not… |

### Safety & Alignment
| Brick | One-liner |
|-------|-----------|
| [Jailbreaks & Adversarial Prompts](jailbreaks-adversarial-prompts.md) | Inputs crafted to bypass safety alignment and elicit refused behavior — spanning hand-written roleplay/persona attacks, optimization-found… |
| [Automated Red-Teaming](automated-red-teaming.md) | Use models, RL, or evolutionary search to automatically generate diverse adversarial prompts that elicit failures (harm, jailbreaks, leakage) at… |
| [Prompt Injection](prompt-injection.md) | Adversarial instructions smuggled into content the LLM *reads* (tool outputs, retrieved docs, web pages, emails, code comments) that the model then… |
| [Sycophancy](sycophancy.md) | The tendency of RLHF-trained models to tell users what they want to hear — agreeing, flattering, conceding to pushback, validating false beliefs —… |
| [Deceptive Alignment / Scheming](deceptive-alignment-scheming.md) | A model that instrumentally behaves aligned during training/eval — to preserve its current goals from gradient updates and to get deployed — while… |
| [Specification Gaming](specification-gaming.md) | A learner satisfies the literal, measurable specification of a task while violating the designer's intent — the umbrella failure of which RL reward… |
| [Machine Unlearning](machine-unlearning.md) | Removing the influence of specific data or capabilities (hazardous knowledge, copyright, PII) from a trained model without full retraining — most… |
| [Watermarking (LLM Output)](watermarking.md) | Embed a statistically detectable signal into generated text by biasing token sampling at decode time (green-list logit boost, or sampling-based… |
| [Safety & Dangerous-Capability Evals](safety-evals.md) | Empirical measurement of risky *capabilities* (CBRN/bio uplift, cyber-offense, autonomous replication, deception) and *propensities* (jailbreak… |
| [Scalable Oversight](scalable-oversight.md) | The umbrella of techniques for supervising models on tasks where humans can't directly judge correctness — by amplifying, decomposing, or… |
| [AI Safety via Debate](ai-safety-via-debate.md) | A scalable-oversight protocol where two AIs argue opposing answers and a weaker (human or model) judge picks the winner — betting that exposing a lie… |
| [Weak-to-Strong Generalization](weak-to-strong-generalization.md) | Fine-tuning a strong pretrained model on labels from a *weaker* supervisor and having it generalize *beyond* the supervisor's accuracy — an empirical… |
| [Refusal & Safety Training](refusal-safety-training.md) | Post-training (SFT + preference optimization) that teaches a model to decline harmful requests — but the refusal behavior is often mediated by a… |
| [Model Organisms of Misalignment](model-organisms-of-misalignment.md) | Deliberately training a model to exhibit a target misalignment (backdoored "sleeper agent," alignment faking, reward hacking, scheming) so the… |
| [Backdoors / Data Poisoning](backdoors-data-poisoning.md) | Hidden trigger→target mappings implanted via poisoned pretraining/fine-tuning data so the model behaves normally (passing evals) except on… |
| [Safety Cases](safety-cases.md) | A structured, evidence-backed argument that deploying a specific model is acceptably safe — borrowed from aviation/nuclear/medical-device assurance —… |
| [Membership Inference & Training-Data Extraction](membership-inference-training-data-extraction.md) | Membership inference (MIA) tests whether a given sample was in the training set; extraction attacks make a model regurgitate verbatim memorized data… |
| [Differential Privacy (DP-SGD)](differential-privacy.md) | Per-example gradient clipping + Gaussian noise on the gradient sum, giving each training example a formal (ε,δ) bound on how much any output can… |

### Multimodal, Embeddings & Retrieval
| Brick | One-liner |
|-------|-----------|
| [CLIP / Contrastive Vision-Language Pretraining](clip-contrastive-vision-language-pretraining.md) | Dual encoders (image tower + text tower) trained on web-scale (image, caption) pairs with a symmetric InfoNCE loss over in-batch positives/negatives… |
| [Vision Transformer (ViT) Patchification](vision-transformer-patchification.md) | Split an image into a grid of non-overlapping P×P patches, flatten and linearly project each into a token embedding, add positional encodings, and… |
| [SigLIP Sigmoid Contrastive Loss](siglip-sigmoid-contrastive-loss.md) | Replace CLIP's softmax/InfoNCE with a per-pair binary sigmoid loss so the objective decouples from global batch size and skips all-pairs… |
| [InfoNCE / Contrastive Loss with Temperature](infonce-contrastive-loss-with-temperature.md) | Cross-entropy over similarity logits that pulls one positive together against many negatives, scaled by a temperature τ that controls how sharply the… |
| [VLM Connector / Projector (LLaVA-style)](vlm-connector-projector.md) | A small trainable module (linear/MLP or cross-attention resampler) that maps a frozen vision encoder's patch embeddings into the LLM's… |
| [Cross-Attention Resampler / Q-Former](cross-attention-resampler-q-former.md) | A small bank of learned query tokens that cross-attend to a large, variable-length set of encoder (usually vision) features and compress them into a… |
| [AnyRes / Dynamic High-Resolution Tiling](anyres-dynamic-high-resolution-tiling.md) | Feed a VLM high-resolution images by splitting them into a grid of native-resolution tiles plus one downsampled global thumbnail, encoding each… |
| [Modality Gap](modality-gap.md) | In jointly trained contrastive spaces (CLIP-style), image and text embeddings collapse into two narrow, well-separated cones on the unit sphere… |
| [Dense Retrieval (Bi-Encoder Embeddings)](dense-retrieval.md) | Encode queries and documents *independently* into fixed-size vectors with a shared (or two-tower) encoder, then retrieve by nearest-neighbor… |
| [RAG (Retrieval-Augmented Generation)](rag.md) | Condition generation on passages fetched at inference from an external corpus, so answers draw on non-parametric, updatable knowledge rather than… |
| [Cross-Encoder Reranking](cross-encoder-reranking.md) | Feed query and candidate together through one transformer with full cross-token attention to emit a single relevance score, then resort the… |
| [ANN Vector Search (HNSW / IVF-PQ)](ann-vector-search.md) | Approximate nearest-neighbor indexes that trade exactness for sublinear query time via navigable small-world graphs (HNSW) or coarse-cluster routing… |
| [Matryoshka Representation Learning (MRL)](matryoshka-representation-learning.md) | Train one embedding so that every leading prefix (first m dims) is itself a usable representation, letting you truncate at serve time to trade… |
| [Late Interaction (ColBERT / ColPali)](late-interaction.md) | Store *per-token* (ColBERT) or *per-patch* (ColPali) embeddings and score a query-document pair by summing the MaxSim — each query token's best match… |
| [Hybrid Search & Reciprocal Rank Fusion](hybrid-search-reciprocal-rank-fusion.md) | Run dense (semantic embedding) and sparse/lexical (BM25, SPLADE) retrieval in parallel, then fuse their ranked lists with Reciprocal Rank Fusion to… |
| [Learned Sparse Retrieval (SPLADE)](learned-sparse-retrieval.md) | Use a transformer's MLM head to predict a sparse, term-weighted vocabulary vector (with learned term expansion) for queries and documents, so… |
| [Hard Negative Mining](hard-negative-mining.md) | Train contrastive retrievers against negatives that are semantically close to the query (high similarity, wrong answer) rather than random in-batch… |
| [Embedding Pooling & Normalization](embedding-pooling-normalization.md) | Collapse a sequence of token embeddings into one fixed vector (mean / CLS / last-token / attention pool) then L2-normalize so that dot product equals… |
| [Large-Batch / Memory-Bank Negatives](large-batch-memory-bank-negatives.md) | Contrastive learning needs many negatives per positive; you get them via huge in-batch negatives (CLIP), a momentum-encoded memory queue decoupled… |
| [Multimodal / Visual Document Embeddings](multimodal-visual-document-embeddings.md) | Embed rendered document pages directly as images (or jointly text+image) with a vision-language model so layout, tables, and figures are retrievable… |

### Agents & Tool Use
| Brick | One-liner |
|-------|-----------|
| [Function / Tool Calling](function-tool-calling.md) | The LLM emits a structured (usually JSON) call to a developer-declared function whose execution result is fed back into context as a new message —… |
| [ReAct (Reason + Act)](react.md) | Interleave chain-of-thought reasoning with tool actions and environment observations in a loop, so each thought conditions the next action and each… |
| [Model Context Protocol (MCP)](model-context-protocol.md) | Open JSON-RPC 2.0 protocol (Anthropic, Nov 2024) standardizing how LLM apps expose and consume tools, resources, and prompts over a client-server… |
| [Constrained / Structured Decoding](constrained-structured-decoding.md) | At each decode step, mask the next-token logits against a grammar (JSON schema, regex, CFG) so only tokens that keep the output on a valid prefix can… |
| [Multi-Agent Orchestration (Orchestrator-Worker)](multi-agent-orchestration.md) | A lead/orchestrator agent decomposes a task, spawns specialized worker subagents that each run in their own context window (parallel or sequential),… |
| [Context Engineering](context-engineering.md) | The discipline of deliberately curating exactly what occupies the finite context window each agent step — system prompt, tools, retrieved facts,… |
| [Context Compaction / Summarization](context-compaction-summarization.md) | When the context window fills, periodically replace older turns and verbose tool outputs with an LLM-generated condensed state (salient facts,… |
| [Agentic Memory Architectures](agentic-memory-architectures.md) | An OS-like memory hierarchy for LLM agents — a small in-context working set (scratchpad) backed by external long-term stores (vector/graph/file/SQL)… |
| [Computer Use / Browser Agents](computer-use-browser-agents.md) | Agents that perceive a GUI as screenshots and/or accessibility/DOM trees and act through low-level mouse/keyboard primitives (click(x,y), type,… |
| [Planning & Task Decomposition](planning-task-decomposition.md) | Explicitly emit a plan or tree/graph of subgoals before (plan-and-solve) or during (interleaved, self-revising) execution, then dispatch them to an… |
| [Self-Reflection / Reflexion](self-reflection-reflexion.md) | The agent verbally critiques its own failed output/trajectory and stores that critique in context for the retry — turning a scalar/environment signal… |
| [Agentic Reinforcement Learning (Tool / Agent RL)](agentic-reinforcement-learning.md) | RL over multi-turn trajectories where the policy interleaves reasoning with real tool calls (search, code execution, browser) and is optimized… |
| [Agentic / Iterative Retrieval](agentic-iterative-retrieval.md) | Turn RAG into a control loop — the model plans, issues multiple queries, reads results, decides whether evidence is sufficient, and reformulates or… |
| [Sub-Agent / Context Isolation](sub-agent-context-isolation.md) | Spawn a fresh-context sub-agent to run a bounded subtask (search, read, tool-loop), returning only a distilled result to the parent — keeping the… |
| [KV-Cache-Aware Agent Design](kv-cache-aware-agent-design.md) | Engineer agent prompts as a stable, append-only token stream so the longest possible prefix hits the provider's prompt cache, turning O(N²) recompute… |
| [Agent Security & the Lethal Trifecta](agent-security-the-lethal-trifecta.md) | A tool-using agent is exploitable for data exfiltration precisely when three capabilities co-occur — access to private data, exposure to untrusted… |

### Evaluation, Uncertainty & Stats
| Brick | One-liner |
|-------|-----------|
| [LLM-as-a-Judge](llm-as-a-judge.md) | Use a strong LLM with a rubric to score or pairwise-rank model outputs as a scalable proxy for human evaluation — the default open-ended-eval… |
| [Judge Bias & Mitigation](judge-bias-mitigation.md) | Systematic distortions in LLM-as-judge scoring — position, verbosity, self-preference, sycophancy, bandwagon — corrected by position-swapping, length… |
| [Bradley-Terry Model](bradley-terry-model.md) | A pairwise-comparison model where P(i beats j) = σ(s_i − s_j), fitting latent quality scores by MLE on win/loss data — the statistical backbone of… |
| [Elo / Online Rating for Model Ranking](elo-online-rating-for-model-ranking.md) | An incremental SGD-style update on Bradley-Terry log-skills — maintain one rating per model and after each pairwise battle nudge winner up / loser… |
| [Benchmark Contamination](benchmark-contamination.md) | Leakage of eval/benchmark examples (or near-duplicates) into pretraining or fine-tuning corpora, inflating scores so they measure memorization rather… |
| [Benchmark Saturation & Dynamic/Private Benchmarks](benchmark-saturation-dynamic-private-benchmarks.md) | Static public benchmarks get topped and contaminated within months, so evaluation moves to held-out private test sets, periodically refreshed live… |
| [pass@k & Self-Consistency Estimation](pass-k-self-consistency-estimation.md) | pass@k estimates the probability at least one of k samples is correct (unbiased estimator from n≥k draws), while self-consistency reports the… |
| [Calibration & ECE](calibration.md) | A model is calibrated when its predicted confidence equals empirical accuracy (of all predictions at 80% confidence, 80% are correct); Expected… |
| [Temperature Scaling](temperature-scaling.md) | A one-parameter post-hoc calibration that divides logits by a single learned scalar T before softmax — fixes over/under-confidence without touching… |
| [Conformal Prediction](conformal.md) | Distribution-free, finite-sample prediction sets with a guaranteed coverage probability — valid under exchangeability only, no assumptions about the… |
| [Prediction-Powered Inference (PPI)](ppi.md) | Estimate a population quantity (mean, regression coefficient, quantile) using a large pool of ML predictions, then debias with a small gold-labeled… |
| [Epistemic vs Aleatoric Uncertainty](epistemic-vs-aleatoric-uncertainty.md) | Decompose predictive uncertainty into epistemic (reducible — model/knowledge uncertainty, shrinks with data) and aleatoric (irreducible — inherent… |
| [Deep Ensembles & MC-Dropout](deep-ensembles-mc-dropout.md) | Approximate the Bayesian predictive distribution by averaging M independently-trained networks (deep ensembles) or M dropout-masked forward passes at… |
| [Semantic Entropy for Hallucination Detection](semantic-entropy-for-hallucination-detection.md) | Detect LLM confabulation by sampling N generations for a prompt, clustering them by bidirectional entailment (same meaning), and computing entropy… |
| [A/B Testing Statistics (Power, MDE, Multiple Comparisons)](a-b-testing-statistics.md) | Sizing experiments so a real effect of size MDE is detectable with power 1−β at level α, shrinking variance via pairing/CUPED, and controlling… |
| [Sequential Testing & Always-Valid p-Values](sequential-testing-always-valid-p-values.md) | Anytime-valid inference (e-values, mSPRT, confidence sequences) where the Type-I error guarantee holds at every sample size simultaneously — so you… |
| [ROC/PR Curves, AUC & Threshold Selection](roc-pr-curves-auc-threshold-selection.md) | Threshold-free summaries of a binary scorer's ranking quality — ROC plots TPR vs FPR, PR plots precision vs recall; AUC is the area under each; the… |
| [Bootstrap & Confidence Intervals for Eval Metrics](bootstrap-confidence-intervals-for-eval-metrics.md) | Resample your eval set with replacement, recompute the metric each time, and read CIs off the resampled distribution — the cheapest way to know… |
| [Long-Context Eval Methodology (Needle-in-Haystack, RULER)](long-context-eval-methodology.md) | Synthetic retrieval/aggregation probes — planting "needle" facts in long filler ("haystack") and varying depth × length — to measure the *effective*… |
| [Perplexity & Bits-Per-Byte](perplexity-bits-per-byte.md) | Perplexity is exp of per-token cross-entropy (the effective branching factor of the next-token distribution); bits-per-byte renormalizes the same… |

### Math & Probability Foundations
| Brick | One-liner |
|-------|-----------|
| [Softmax](softmax.md) | Converts a vector of real-valued logits to a probability distribution; temperature τ controls the sharpness. |
| [Layer Normalization](layer-norm.md) | Normalize each token's activation vector over the feature dimension to zero mean / unit variance, then apply a learned per-feature scale γ and shift… |
| [KL Divergence](kl-divergence.md) | Asymmetric measure of how different distribution Q is from distribution P; appears in RLHF, VAEs, conformal prediction, and generalization bounds. |
| [Cross-Entropy](cross-entropy.md) | Expected negative log-likelihood of data under a model's predicted distribution; the loss minimized by classification and next-token prediction;… |
| [Log-Derivative Trick (REINFORCE / Score Function)](log-derivative-trick.md) | ∇θ E_{x∼p_θ}[f(x)] = E_{x∼p_θ}[f(x)·∇θ log p_θ(x)] — turns the gradient of an expectation into an expectation of a gradient you can Monte Carlo… |
| [Reparameterization Trick](reparameterization.md) | Move the randomness of a sample out of the learnable parameters into fixed external noise — z = μ + σ⊙ε with ε∼N(0,I) — so ∇θ E[f(z)] becomes an… |
| [SVD & Low-Rank Approximation](svd.md) | Every matrix factors as M = UΣVᵀ (orthonormal U, V; nonnegative diagonal Σ); truncating to the top-k singular values gives the *provably optimal*… |
| [Eigendecomposition](eigendecomposition.md) | A = QΛQ⁻¹ factors a square matrix into its invariant directions (eigenvectors, columns of Q) and how much it stretches each (eigenvalues, diagonal of… |
| [Matrix Rank](matrix-rank.md) | The number of linearly independent rows/columns of a matrix = its effective dimensionality = the count of nonzero singular values; the whole reason… |
| [Jacobian](jacobian.md) | The matrix of all first-order partials J_{ij} = ∂f_i/∂x_j of a vector map f: ℝⁿ → ℝᵐ; it is the best local linear approximation of f, and the chain… |
| [Byte-Pair Encoding (BPE)](bpe.md) | A greedy tokenizer-training algorithm that starts from bytes/characters and iteratively merges the most frequent adjacent symbol pair into a new… |
| [Backpropagation / Reverse-Mode Autodiff](backpropagation.md) | The chain rule run backward through a computation graph — one forward pass caches activations, one backward pass propagates the loss gradient to… |
| [Batch Normalization](batch-norm.md) | Normalize each activation channel across the batch dimension using mini-batch mean/variance during training and a running EMA at inference; the… |
| [Gumbel-Softmax / Straight-Through Estimator](gumbel-softmax-straight-through-estimator.md) | Two tricks for backprop through discrete sampling — Gumbel-Softmax relaxes a categorical sample into a differentiable softmax over noised logits;… |
| [Maximum Mean Discrepancy / Wasserstein Distance](maximum-mean-discrepancy-wasserstein-distance.md) | Two distribution distances that, unlike KL, stay finite and informative for disjoint-support distributions — MMD is a kernel two-sample statistic,… |

### Causal ML
| Brick | One-liner |
|-------|-----------|
| [Potential Outcomes (Neyman–Rubin)](potential-outcomes.md) | Define causal effects as contrasts of per-unit counterfactual outcomes Y(1),Y(0); only one is ever observed (the "fundamental problem of causal… |
| [DAGs & Do-Calculus](do-calculus.md) | Pearl's framework — a causal DAG plus three rewrite rules that turn an interventional query P(Y / do(X)) into estimable observational quantities; the… |
| [Backdoor Criterion & Confounding Adjustment](backdoor-criterion-confounding-adjustment.md) | A graphical test for "what to control for" — a covariate set Z that blocks every back-door path from X to Y (and contains no descendant of X)… |
| [Instrumental Variables (IV)](instrumental-variables.md) | When ignorability fails (unobserved confounder U affects both treatment D and outcome Y), recover the causal effect using an instrument Z that… |
| [Frontdoor Criterion & Mediation Analysis](frontdoor-criterion-mediation-analysis.md) | Identify P(Y / do(X)) by routing the effect through a fully-mediating variable M — using M's own (unconfounded) front-door path — when an unobserved… |
| [Double / Debiased Machine Learning (DML)](double-debiased-machine-learning.md) | Estimate a low-dimensional causal effect at √n rate by plugging flexible ML nuisance estimates into a Neyman-orthogonal moment and cross-fitting —… |
| [Propensity Scores & IPW](propensity-scores-ipw.md) | The propensity score e(x)=P(T=1∣X=x) is the coarsest balancing score — conditioning on it suffices to remove confounding under ignorability +… |

### Foundational & Cross-Cutting
| Brick | One-liner |
|-------|-----------|
| [Alignment (Overview)](alignment.md) | The problem of making capable models reliably pursue intended goals/values — in 2024-26 practice a pipeline of preference learning (RLHF/DPO),… |
| [Causal Representation Learning (CRL)](causal-representation-learning.md) | Recover latent causal variables z and their causal graph G from high-dimensional observations x = g(z) (pixels, text), so the learned representation… |
| [Distributed Training (Parallelism Overview)](distributed-training.md) | Frontier training composes orthogonal parallelism axes — data, tensor, pipeline, sequence/context, and expert — into a 3D/4D mesh, each axis trading… |
| [Inference & Serving](inference-and-serving.md) | The decode-time stack — compute-bound *prefill* followed by memory-bound *decode*, scheduled via continuous batching over a paged KV cache with… |
| [Multilayer Perceptron (MLP)](mlp.md) | Stacked affine maps interleaved with pointwise nonlinearities — the universal-approximator primitive and the position-wise FFN that holds most of a… |
| [Training Stability](training-stability.md) | The set of tricks that keep large-scale (especially low-precision, high-LR) training from diverging — LR warmup, gradient clipping, QK-norm, z-loss,… |

---
## How to add a brick

Copy the template (see any existing brick): `# Title` · `**One-liner:**` · formula/definition · where it appears · common mistake · `## See also` with `[[slug]]` links. Name it `kebab-case.md`, add a row above, tick it off in [topic-map.md](topic-map.md).
