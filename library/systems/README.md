# Systems

The infrastructure that makes frontier-scale training and serving physically possible.

## Topics

| Topic | File | One-line summary |
|-------|------|-----------------|
| GPU Fundamentals | [gpu-fundamentals.md](gpu-fundamentals.md) | Memory hierarchy, parallelism, why everything is memory-bound |
| Distributed Training | [distributed-training.md](distributed-training.md) | Data parallel, tensor parallel, pipeline parallel — ZeRO, FSDP, Megatron |
| Attention Efficiency | [attention-efficiency.md](attention-efficiency.md) | FlashAttention — IO-awareness as the core idea |
| Inference & Serving | [inference-and-serving.md](inference-and-serving.md) | KV cache, PagedAttention, speculative decoding, quantization |
