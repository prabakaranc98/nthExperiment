# QLoRA

**One-liner:** LoRA adapters trained on top of a frozen 4-bit NF4-quantized base, with double quantization and paged optimizers — fine-tunes a 65B model on one 48GB GPU at ~16-bit accuracy.

## The mechanism

The base weights W are stored once in 4-bit NF4 and never updated. The forward pass dequantizes a tile of W to bf16 on the fly, then computes:

  Y = dequant(W_NF4) · X  +  (α/r) · B · A · X

Only the LoRA adapters A, B (bf16) carry gradients. Backprop flows through dequant(W) into A and B but **not** into W — the 4-bit weights are constant, so no quantization-aware training and no STE on the base. Memory is the 4-bit base (read-only) + small bf16 adapters + bf16 optimizer states for the adapters only.

## The three ingredients

- **NF4 (4-bit NormalFloat):** an information-theoretically optimal quantile-quantization for ~N(0, σ²) weights. The 16 codebook levels are set to the quantiles of a standard normal, so each bucket holds equal probability mass. Block-wise (e.g. block size 64) with per-block scales.
- **Double quantization:** quantize the per-block FP32 scales themselves (8-bit, blocks of 256). Saves ~0.37 bits/param — meaningful at 65B.
- **Paged optimizers:** optimizer state lives in unified memory and is paged to CPU RAM on gradient-checkpoint spikes, preventing OOM during long sequences (NVIDIA unified-memory paging, like CPU page faults for the GPU).

## Where it appears

- QLoRA (Dettmers et al., 2023, [arXiv 2305.14314](https://arxiv.org/abs/2305.14314)) — the original; Guanaco models matched ChatGPT-era quality fine-tuned on one GPU
- bitsandbytes + HuggingFace PEFT — the default OSS recipe for single-GPU instruction tuning and DPO/GRPO post-training
- Long-context and small-lab RLHF — the only practical way to do gradient-based post-training of 70B+ on consumer/prosumer hardware

## Common mistake

Thinking QLoRA quantizes the *trained* model for deployment, or that it does quantization-aware training. It does neither: NF4 is a frozen, lossy storage format for the base used **only to save memory during fine-tuning**. The adapters train in bf16; for inference you typically dequantize and merge (or re-quantize separately). The 4-bit base is constant — gradients never touch it.

## See also
- [[lora]] — QLoRA is LoRA with the base swapped for a 4-bit frozen copy
- [[quantization]] — NF4 is the weight-only quant scheme; QLoRA is weight-only (dequant before matmul)
- [[gradient-checkpointing]] — paged optimizers exist to absorb the memory spikes checkpointing creates
