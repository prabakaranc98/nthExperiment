# Topic 09 · Algorithms & Systems for AI

*The systems that make scale tractable — parallelism, precision, kernels, serving, and the retrieval/quantization algorithms underneath.*

**Papers:** 20 · **Pace:** ~10 days at 2/day

---

## Paper Log

| # | Paper | Authors | Year | Status | Note | Blog | Exp |
|---|-------|---------|------|--------|------|------|-----|
| 1 | Distilling the Knowledge in a Neural Network | Hinton, Vinyals & Dean | 2015 | queued | Soft targets transfer the function, not the weights | — | — |
| 2 | Mixed Precision Training | Micikevicius et al. | 2017 | queued | FP16 + loss scaling; basis of modern throughput | — | — |
| 3 | PyTorch | Paszke et al. | 2019 | queued | Define-by-run autodiff as the execution model | — | — |
| 4 | GPipe | Huang et al. | 2018 | queued | Pipeline parallelism with micro-batching | — | — |
| 5 | Megatron-LM | Shoeybi et al. | 2019 | queued | Tensor (intra-layer) parallelism | — | — |
| 6 | ZeRO | Rajbhandari et al. | 2020 | queued | Sharding optimizer/gradient/parameter state across ranks | — | — |
| 7 | Efficient Large-Scale Training (3D parallelism) | Narayanan et al. | 2021 | queued | Composing tensor + pipeline + data parallelism | — | — |
| 8 | Switch Transformers | Fedus, Zoph & Shazeer | 2021 | queued | MoE routing for cheap scale | — | — |
| 9 | Reducing Activation Recomputation | Korthikanti et al. | 2022 | queued | Selective checkpointing; the memory lever | — | — |
| 10 | FlashAttention | Dao et al. | 2022 | queued | IO-aware exact attention | — | — |
| 11 | FlashAttention-2 | Dao | 2023 | queued | Better parallelism and work partitioning | — | — |
| 12 | LLM.int8() | Dettmers et al. | 2022 | queued | Outlier-aware 8-bit inference without quality loss | — | — |
| 13 | GPTQ | Frantar et al. | 2022 | queued | Accurate post-training quantization via second-order information | — | — |
| 14 | QLoRA | Dettmers et al. | 2023 | queued | 4-bit base + low-rank adapters; fine-tuning on one GPU | — | — |
| 15 | FP8-LM | Peng et al. | 2023 | queued | 8-bit floating point for training | — | — |
| 16 | vLLM / PagedAttention | Kwon et al. | 2023 | queued | KV-cache paging for high-throughput serving | — | — |
| 17 | Speculative Decoding | Leviathan et al. | 2023 | queued | Draft-and-verify; latency without quality loss | — | — |
| 18 | FAISS | Johnson, Douze & Jégou | 2017 | queued | GPU approximate nearest-neighbor search | — | — |
| 19 | HNSW | Malkov & Yashunin | 2016 | queued | Graph index behind most vector databases | — | — |
| 20 | Triton | Tillet, Kung & Cox | 2019 | queued | Writing fused GPU kernels without deep CUDA | — | — |

---

## Synthesis Notes
## Blog Post
## Experiments
