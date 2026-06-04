# Systems & Efficiency

Distributed training, GPU kernels, inference optimization, and the infrastructure that makes frontier-scale models possible.

For the underlying concepts (attention, KV cache, MoE, quantization math), see the concept-library index: [`../bricks/README.md`](../bricks/README.md).

---

## Start here

The shortest path from "I train models" to "I understand what the hardware is doing." Read these three in order, then branch into the sections below.

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Making Deep Learning Go Brrr (Horace He)](https://horace.io/brrr_intro.html) | The memory-bound vs. compute-bound mental model for ML. The single best framing for "why is my code slow" — roofline thinking made practical. Still the canonical starting point. | 🟡 |
| [How to Scale Your Model (Google DeepMind, 2024)](https://jax-ml.github.io/scaling-book/) | The "TPU performance handbook." A first-principles, arithmetic-driven walk through rooflines, sharding, and parallelism — the clearest modern systems text. Concepts transfer directly to GPUs. | 🟡 |
| [How to Train Really Large Models on Many GPUs? (Lilian Weng)](https://lilianweng.github.io/posts/2021-09-25-train-large/) | The complete map of data, tensor, and pipeline parallelism plus ZeRO, at the right level of abstraction. Read before any distributed-training work. | 🟡 |

---

## Distributed training & parallelism

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [The Ultra-Scale Playbook (Hugging Face, 2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | The current best end-to-end guide to training on GPU clusters: 5D parallelism, ZeRO, activation recomputation, overlap, and the memory math — with runnable code from the `nanotron` repo. Supersedes most older parallelism explainers. | 🟡 |
| [How to Scale Your Model (Google DeepMind, 2024)](https://jax-ml.github.io/scaling-book/) | First-principles treatment of why each parallelism strategy exists and when it wins. Best for building intuition rather than copying recipes. | 🟡 |
| Megatron-LM / Megatron-Core (NVIDIA) | Reference implementation of tensor + sequence + pipeline parallelism that most production training stacks borrow from. Read the source and the original papers (tensor parallelism, 2019; sequence parallelism + selective recompute, 2022) for the canonical techniques. | 🔴 |

---

## Inference optimization & serving

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Large Transformer Model Inference Optimization (Lilian Weng)](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | The survey that ties together KV cache, speculative decoding, quantization, and PagedAttention. Still the best one-page orientation to the whole toolkit. | 🟡 |
| [KV Caching Explained (Hugging Face)](https://huggingface.co/blog/not-lain/kv-caching) | How the KV cache works, why it grows linearly with sequence length, and why it is the dominant memory cost at serving time. | 🟢 |
| [Unlocking Asynchronicity in Continuous Batching (Hugging Face)](https://huggingface.co/blog/continuous_async) | How continuous batching lifts inference throughput — the core idea that vLLM and friends are built on. | 🟡 |
| vLLM docs & PagedAttention paper (2023) | PagedAttention treats the KV cache like OS virtual memory to cut fragmentation; the project is now the de-facto open serving engine. Read the docs for prefix caching, chunked prefill, and disaggregated prefill/decode. | 🟡 |
| SGLang (RadixAttention) | Serving engine whose RadixAttention shares KV-cache prefixes across requests via a radix tree — large wins for agentic and few-shot workloads with repeated context. Read the project docs/paper. | 🔴 |

---

## GPU kernels: Triton & CUDA

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Triton tutorials (official)](https://triton-lang.org/main/getting-started/tutorials/index.html) | Writing custom GPU kernels in Python — vector add, fused softmax, matmul, then fused attention. The standard on-ramp to kernel writing. | 🔴 |
| [GPU MODE (formerly CUDA MODE)](https://github.com/gpu-mode/lectures) | Community lecture series and code on writing fast kernels — Triton, CUDA, profiling, quantization, and FlashAttention internals. The most active modern resource for learning GPU programming. Lectures on YouTube, code on GitHub. | 🔴 |
| [CUDA C++ Programming Guide (NVIDIA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) | The authoritative reference on the CUDA model — thread/memory hierarchy, async copy, tensor cores, performance guidelines. Reference, not tutorial. | 🔴 |
| FlashAttention papers (v1 2022, v2 2023, v3 2024) | The IO-aware attention kernel that made long context tractable; v3 targets Hopper async/FP8. Read v2 for the algorithm, v3 for current hardware-specific tricks. | 🔴 |

---

## Quantization & low-precision

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Large Transformer Model Inference Optimization (Lilian Weng)](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | Survey-level entry point covering PTQ, QAT, and INT8/INT4 alongside the rest of the inference stack. | 🟡 |
| [A Visual Guide to Quantization (Maarten Grootendorst, 2024)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) | The clearest modern explainer of the quantization landscape — symmetric/asymmetric, GPTQ, GGUF, AWQ, and 1-bit/BitNet ideas — built around diagrams. | 🟢 |
| GPTQ, AWQ, and SmoothQuant papers | The three post-training-quantization methods you will actually encounter in deployed stacks. Read them to know which artifact format and accuracy/latency trade-off you are buying into. | 🔴 |

---

## Profiling & performance

| Resource | What it is + why it's worth your time | Level |
|----------|---------------------------------------|-------|
| [Profiling in PyTorch (Hugging Face)](https://huggingface.co/blog/torch-profiler) | How to use the PyTorch Profiler to find where training time actually goes. Do this before optimizing anything. | 🟡 |
| [Mastering Tensor Dimensions in Transformers (Hugging Face)](https://huggingface.co/blog/not-lain/tensor-dims) | How to track tensor shapes through a forward pass — essential for debugging kernels and implementing models from scratch. | 🟢 |
| PyTorch `torch.compile` docs & profiler trace viewer | The default path to fusion and graph capture in modern PyTorch; pair with the Perfetto/Chrome trace viewer to read kernel timelines. Reference docs. | 🟡 |

---

### Legend

🟢 accessible · 🟡 intermediate · 🔴 advanced / reference. Where a URL is omitted, the source is named so you can find the current canonical version — implementations and docs in this area move fast.
