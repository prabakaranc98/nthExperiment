# Systems & Efficiency — Revision Sheet

One page on the hardware reality that shapes every frontier training and serving decision.

## The GPU mental model

```
Compute (Tensor Cores)  >>  Memory bandwidth (HBM)  >>  Interconnect (NVLink/IB)
   ~1000 TFLOP/s (BF16)      ~3 TB/s (H100)            ~900 GB/s NVLink / ~50 GB/s IB
```

- **Almost everything is memory-bound, not compute-bound.** The lever is moving fewer bytes, not doing fewer FLOPs.
- **Arithmetic intensity** = FLOPs / bytes moved. The roofline says: performance = min(peak FLOP/s, bandwidth × intensity). Low intensity → bandwidth-bound.
- Memory hierarchy: registers → SRAM (shared mem, ~20 MB, ~20 TB/s) → HBM (40–80 GB, ~3 TB/s) → host. Keep hot data as high as possible.

## Parallelism (how to scale past one GPU)

| Axis | Splits | Comm | Use when |
|------|--------|------|----------|
| **Data (DP)** | the batch | all-reduce grads | always, the default |
| **ZeRO / FSDP** | optimizer/grad/param state across DP ranks | all-gather params on demand | model state doesn't fit |
| **Tensor (TP)** | weight matrices within a layer | all-reduce per layer (chatty → keep intra-node) | layer too big for one GPU |
| **Pipeline (PP)** | layers into stages | point-to-point activations | depth too big; mind the bubble |
| **Sequence/Context (CP)** | the sequence dim | ring KV exchange | very long context |
| **Expert (EP)** | MoE experts across devices | all-to-all token routing | MoE models |

Frontier runs combine these (**3D/4D parallelism**): e.g. TP within a node, PP across nodes, DP on top, EP for experts.

## Memory: what eats your VRAM

```
Params (P) + Gradients (P) + Optimizer state (2P for Adam m,v) + Activations
≈ 16·P bytes (FP32 Adam, mixed precision)  →  ZeRO shards the 16·P term
Activations scale with batch × seq × layers  →  gradient checkpointing trades compute to shrink them
```

## Efficiency techniques (the toolbox)

| Technique | Idea | Win |
|-----------|------|-----|
| **Mixed precision** | BF16 compute + FP32 master | ~2× throughput/memory |
| **FP8 / MXFP4** | block-scaled low precision (Blackwell) | another ~2× |
| **FlashAttention** | tile + online softmax, avoid N² HBM | 2–4× attention |
| **Gradient checkpointing** | recompute activations in backward | big activation-memory cut |
| **Fused kernels (Triton/CUDA)** | one launch, stay in SRAM | kills launch + HBM overhead |

## Inference & serving

- **Two phases:** *prefill* (compute-bound, parallel over prompt) vs *decode* (memory-bound, one token at a time, KV-cache reads dominate).
- **KV cache** is the decode bottleneck: size = 2 · layers · heads · head_dim · seq · batch · dtype. GQA/MLA shrink it.
- **PagedAttention (vLLM)** — virtual-memory-style KV paging → high batch occupancy.
- **Continuous batching** — fill freed slots mid-flight instead of static batches.
- **Speculative decoding** — draft k tokens cheap, verify in one target pass; same output distribution.
- **Quantization** (GPTQ/AWQ, INT4/FP8) — smaller weights, faster memory reads.

## See also
- [[roofline]] · [[distributed-training]] · [[zero]] · [[fsdp]] · [[tensor-parallel]] · [[flash-attention]] · [[kv-cache]] · [[speculative-decoding]] · [[quantization]] · [[mixed-precision]] · [[gradient-checkpointing]] · [[inference-and-serving]]
