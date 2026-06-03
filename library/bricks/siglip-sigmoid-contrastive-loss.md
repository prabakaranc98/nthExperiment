# SigLIP Sigmoid Contrastive Loss

**One-liner:** Replace CLIP's softmax/InfoNCE with a per-pair binary sigmoid loss so the objective decouples from global batch size and skips all-pairs normalization — the de-facto vision encoder for modern VLMs.

## The formula

For a batch of N image-text pairs, score every pair with cosine similarity scaled by a learnable temperature t (initialized log t) plus a learnable bias b: z_ij = t · x_i·y_j + b. Each of the N² pairs is an independent binary classification — diagonal is positive (label +1), off-diagonal negative (label −1):

L = −(1/N) Σ_i Σ_j log σ( label_ij · (t · x_i·y_j + b) )

where label_ij = +1 if i=j else −1, and σ is the logistic sigmoid. No row/column softmax, no need to gather all logits — the loss factorizes per pair.

The learnable bias b matters: at init the N²−N negatives swamp the N positives, so b is initialized large-negative (~ −10) to counteract the massive negative imbalance.

## Why it scales

InfoNCE needs a softmax over the full batch (an all-gather of every embedding across devices). Sigmoid loss is a sum of independent terms, so the chunked/blockwise implementation computes the loss with only a ring-style pass that swaps small embedding blocks between devices — memory and communication are O(N/devices), not O(N²) materialized globally. SigLIP found performance saturates around batch 32k; you do NOT need CLIP's 100k+ batches.

## Where it appears

- **SigLIP (Zhai et al., ICCV 2023)** — the original sigmoid loss; matched CLIP quality at far smaller batch sizes
- **SigLIP 2 (2025)** — adds captioning, self-distillation, dense (SILC/TIPS-style) objectives; the default open vision encoder
- **PaliGemma, Gemma 3 vision, Idefics, many 2024-2026 VLMs** — SigLIP/SigLIP-2 is the frozen or lightly-tuned image tower behind the projector into the LLM
- **Locked-image tuning / LiT-style setups** — sigmoid loss for cheap contrastive fine-tuning without giant batches

## Common mistake

Thinking the win is just "sigmoid > softmax" as a loss. The real point is decoupling the loss from batch size: InfoNCE's denominator couples every example to every other, forcing huge synchronized batches; the sigmoid factorizes per pair so accuracy holds at modest batch sizes. Also: forgetting the learnable bias b — without the negative init, training is dominated by easy negatives and underperforms.

## See also
- [[clip-contrastive-vision-language-pretraining]] — the softmax/InfoNCE predecessor SigLIP replaces
- [[infonce-contrastive-loss-with-temperature]] — the all-pairs-normalized objective and its batch-size coupling
- [[vlm-connector-projector]] — SigLIP is the image tower feeding the projector in modern VLMs
