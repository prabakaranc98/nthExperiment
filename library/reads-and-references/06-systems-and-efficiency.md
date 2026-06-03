# Systems & Efficiency

Distributed training, GPU kernels, inference optimization, and the infrastructure that makes frontier-scale models possible.

---

## Distributed training (Lilian Weng)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/) | The complete guide to data parallelism, tensor parallelism, pipeline parallelism, and ZeRO — explained at the right level of abstraction. Read before any distributed training work. | 🟡 |
| [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | KV cache, speculative decoding, quantization, PagedAttention — the full inference optimization toolkit. | 🟡 |

---

## Profiling and performance

| Resource | Why read it | Level |
|----------|-------------|-------|
| [HuggingFace: Profiling in PyTorch](https://huggingface.co/blog/torch-profiler) | How to use PyTorch Profiler to find where your training time is actually going. Essential before any optimization. | 🟡 |
| [Making Deep Learning Go Brrr (Horace He)](https://horace.io/brrr_intro.html) | The memory-bound vs. compute-bound mental model applied to ML. Why fusing operations matters. The roofline model made practical. | 🟡 |

---

## KV cache and serving

| Resource | Why read it | Level |
|----------|-------------|-------|
| [HuggingFace: KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) | How the KV cache works, why it grows linearly with sequence length, and why it's the main memory bottleneck. | 🟢 |
| [HuggingFace: Unlocking Asynchronicity in Continuous Batching](https://huggingface.co/blog/continuous_async) | How continuous batching improves inference throughput — the key idea behind vLLM. | 🟡 |

---

## Triton and kernel writing

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Triton tutorials (official)](https://triton-lang.org/main/getting-started/tutorials/index.html) | Writing custom GPU kernels in Python — vector addition, fused softmax, matrix multiplication. Start here for kernel writing. | 🔴 |
| [CUDA Programming Model (NVIDIA docs)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) | The authoritative source on CUDA — thread hierarchy, memory model, performance guidelines. Reference, not tutorial. | 🔴 |

---

## Architecture and tensor dimensions

| Resource | Why read it | Level |
|----------|-------------|-------|
| [HuggingFace: Mastering Tensor Dimensions in Transformers](https://huggingface.co/blog/not-lain/tensor-dims) | How to track tensor shapes through a transformer forward pass — essential for debugging and implementing from scratch. | 🟢 |

---

## Quantization

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng: Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | Covers quantization (PTQ, QAT, INT8/INT4) alongside other inference techniques. The survey-level starting point. | 🟡 |
