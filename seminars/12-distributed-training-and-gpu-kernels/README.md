# Distributed Training and GPU Kernels

The infrastructure layer of modern AI. Training at scale requires understanding parallelism strategies, memory optimization, and low-level compute — CUDA, Triton, and the systems that make large models possible.

## Focus Areas
- Data parallelism, model parallelism, pipeline parallelism, tensor parallelism
- ZeRO optimizer and memory-efficient training
- Mixed precision training (FP16, BF16, FP8)
- Gradient checkpointing and activation recomputation
- CUDA programming model: threads, blocks, warps, memory hierarchy
- Triton: writing custom GPU kernels in Python
- FlashAttention and fused kernels
- Communication primitives: all-reduce, ring all-reduce

## Paper Log

| Title | Authors | Year | Status | Notes |
|-------|---------|------|--------|-------|
| Megatron-LM | Shoeybi et al. | 2019 | queued | Tensor parallelism |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al. | 2019 | queued | |
| FlashAttention | Dao et al. | 2022 | queued | IO-aware exact attention |
| FlashAttention-2 | Dao | 2023 | queued | |
| GPipe: Efficient Training using Pipeline Parallelism | Huang et al. | 2018 | queued | |
| Efficient Large Scale Language Modeling (FairSeq) | Artetxe et al. | 2021 | queued | |
