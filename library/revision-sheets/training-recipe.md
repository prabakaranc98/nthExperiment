# Frontier LLM Training Recipe — Revision Sheet

## The full pipeline

```
1. DATA CURATION
   Web crawl → quality filter → dedup (MinHash/semantic)
   → domain mixing (DoReMi-style weights)
   → tokenize (BPE, 32K–128K vocab)
   → 10T–15T tokens

2. PRETRAINING
   Next-token prediction, cross-entropy loss
   FP8/BF16 mixed precision
   Distributed: data parallel + tensor parallel + pipeline parallel (3D)
   ZeRO / FSDP for optimizer state sharding
   Cosine LR schedule with warmup
   ~10²³–10²⁵ FLOPs total compute

3. SUPERVISED FINE-TUNING (SFT)
   (instruction, response) pairs, ~10K–1M examples
   Chat template, packing
   Short: 1–3 epochs, lower LR than pretraining

4. PREFERENCE OPTIMIZATION
   DPO (offline): train on (preferred, rejected) pairs
   KL penalty to stay close to SFT model
   Or RLHF: reward model + PPO (more expensive)

5. REASONING RL (if applicable)
   GRPO / RLVR on verifiable tasks (math, code)
   Group-relative advantage (no critic needed)
   Builds reasoning through self-play on verifiable rewards

6. DEPLOYMENT
   Quantize (GPTQ / AWQ, INT4/FP8)
   Paged KV cache (vLLM)
   Speculative decoding (draft model)
   Continuous batching
```

## Key efficiency techniques

| Technique | What it does | Gain |
|-----------|-------------|------|
| FP8 training | Half the memory of BF16 | ~2× throughput |
| FlashAttention | Fuse attention ops, avoid HBM writes | ~2–4× for attention |
| Gradient checkpointing | Recompute activations to save memory | Trade compute for memory |
| ZeRO-3 | Shard optimizer + gradients + params | Scales to arbitrary size |
| μP | Hyperparameters transfer from small models | Cheaper ablations |

## The compute-optimal tradeoff (Chinchilla)

For a compute budget C FLOPs:
- Compute-optimal: N ~ C^0.5 params, D ~ C^0.5 tokens (roughly)
- **Inference-optimal**: train smaller model on *more* tokens than compute-optimal
  (most 2024+ models are deliberately "over-trained" for better inference cost)

## Costs (rough, 2024–2025)

| Model | Approx. training cost |
|-------|----------------------|
| DeepSeek-V3 671B | ~$5.6M (2.79M H800-hours) |
| Llama 3.1 405B | ~$10–20M (estimated) |
| GPT-4 | ~$50–100M (estimated) |
