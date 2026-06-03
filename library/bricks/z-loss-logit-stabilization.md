# Z-Loss & Logit Stabilization

**One-liner:** A family of auxiliary regularizers — output/router z-loss penalizing the softmax log-partition, attention-logit soft-capping, and embedding/logit scaling — that keep softmax logits bounded so the partition function stays well-conditioned, preventing roundoff-driven divergence and loss spikes in large-scale pretraining.

## The formula / definition

Z-loss penalizes the squared log-normalizer of a softmax. For logits z over a vocab/expert set, with Z = sum_j exp(z_j):

  L_z = c_z * (log Z)^2 = c_z * (logsumexp(z))^2,  typically c_z ~= 1e-4

added to the main cross-entropy. It pushes log Z toward 0 (i.e., Z toward 1), which keeps individual logits from drifting large in absolute scale and keeps exp() inside fp16/bf16 range. Two main sites:
- **Output z-loss** (PaLM): on the final vocabulary softmax — stops the LM head logits from blowing up.
- **Router z-loss** (ST-MoE): on the MoE gating logits — the dominant stabilizer for sparse MoE training.

Attention-logit **soft-capping** (Gemma 2) bounds pre-softmax scores instead of penalizing them:

  z <- cap * tanh(z / cap)   (cap = 50 for attn, 30 for final logits)

QK-norm achieves similar stabilization structurally by L2-normalizing q,k before the dot product. Embedding scaling (multiply embeddings by sqrt(d_model)) sets the initial logit scale sensibly.

## Where it appears

- **PaLM (2022)** — introduced output z-loss with c_z=1e-4; cited as essential for stable bf16 training at 540B.
- **ST-MoE / GLaM / Switch** — router z-loss is the key trick that makes large sparse-expert training not diverge.
- **Gemma 2 (2024)** — uses tanh soft-capping on attention and final logits instead of z-loss (note: incompatible with FlashAttention, so often dropped at inference).
- **Baichuan, Chameleon, OLMo** — z-loss and/or QK-norm reported as instability fixes; Chameleon used QK-norm + reordered norm for early-fusion multimodal stability.

## Common mistake

Thinking z-loss is just generic weight regularization or label smoothing. It does not regularize predictions or weights — it specifically constrains the log-partition log Z so the softmax denominator stays near 1, attacking the numerical/conditioning failure mode (logit norm growth) that causes spikes. Also: soft-capping and z-loss target different surfaces (bounding vs. penalizing), and capping breaks fused-softmax kernels.

## See also
- [[loss-spikes-training-instability]] — the failure z-loss is designed to prevent
- [[qk-normalization]] — structural alternative that bounds attention logits directly
- [[softmax-bottleneck-logit-cap-final-layer-tying]] — related final-layer logit-scale interventions
