# Systems

The infrastructure that makes frontier-scale training and serving physically possible.

## Topics

| Topic | File | One-line summary |
|-------|------|-----------------|
| GPU Fundamentals | [gpu-fundamentals.md](gpu-fundamentals.md) | Memory hierarchy, parallelism, why everything is memory-bound |
| Distributed Training | [distributed-training](../bricks/distributed-training.md) | Data parallel, tensor parallel, pipeline parallel — ZeRO, FSDP, Megatron |
| Attention Efficiency | [flash-attention](../bricks/flash-attention.md) | FlashAttention — IO-awareness as the core idea |
| Inference & Serving | [inference-and-serving](../bricks/inference-and-serving.md) | KV cache, PagedAttention, speculative decoding, quantization |
