# Inference, Scaling and Systems for AI/ML

The systems that take trained models and make them fast, cheap, and deployable at scale — from quantization to speculative decoding to inference-time compute scaling.

## Focus Areas
- Quantization: PTQ, QAT, GPTQ, AWQ, FP8
- Speculative decoding and draft models
- KV cache: structure, compression, paged attention
- Inference-time compute scaling: best-of-N, MCTS, process reward models
- Batching strategies: continuous batching, chunked prefill
- Serving frameworks: vLLM, TensorRT-LLM, SGLang
- Test-time training and adaptation
- Sparsity: MoE, sparse attention, pruning

## Paper Log

| Title | Authors | Year | Status | Notes |
|-------|---------|------|--------|-------|
| Efficient Memory Management for LLM Serving (PagedAttention / vLLM) | Kwon et al. | 2023 | queued | |
| Speculative Decoding | Chen et al. | 2023 | queued | Faster inference with draft models |
| GPTQ: Post-training Quantization | Frantar et al. | 2022 | queued | 4-bit quantization |
| Scaling LLM Test-Time Compute | Snell et al. | 2024 | queued | Inference-time scaling |
| Mixture of Experts (Switch Transformer) | Fedus et al. | 2021 | queued | |
| AWQ: Activation-Aware Weight Quantization | Lin et al. | 2023 | queued | |
