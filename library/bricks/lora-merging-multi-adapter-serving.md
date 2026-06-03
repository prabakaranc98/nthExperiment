# LoRA Merging & Multi-Adapter Serving

**One-liner:** Either fold ΔW=(α/r)BA back into W₀ for zero-overhead single-tenant inference, or keep adapters unmerged and hot-swap/batch hundreds of them off one shared base (S-LoRA, Punica) for multi-tenant serving.

## The two regimes

**Merge (single adapter, max throughput):**
W_merged = W₀ + (α/r)·BA — done once, offline. Inference is identical to the base model: no extra FLOPs, no extra memory, no per-token adapter matmul. Irreversible per checkpoint; to switch adapters you must un-merge (W₀ = W_merged − (α/r)BA) or reload.

**Unmerged (many adapters, one base):** keep W₀ shared and resident; for request i with adapter (Aᵢ,Bᵢ):
  y = W₀x + (αᵢ/rᵢ)·Bᵢ(Aᵢx)
The base matmul W₀x is batched across all requests; only the tiny low-rank term is per-adapter. This is the key win — a single base model serves thousands of fine-tunes at near-base throughput.

## The serving trick (batched heterogeneous LoRA)

A naive loop over adapters serializes the GEMMs. **Punica** introduces **SGMV** (Segmented Gather Matrix-Vector): a single fused kernel applies different (Aᵢ,Bᵢ) to different segments of a batch in one launch. **S-LoRA** adds **Unified Paging** — store adapter weights and KV cache in one paged pool, swapping adapters between host/GPU memory at request granularity, enabling thousands of adapters with one base. Both decouple the O(1) base cost from O(rank) per-adapter cost.

## Where it appears

- **S-LoRA** (Sheng et al., 2023) — unified paging + custom CUDA kernels; serves thousands of LoRA adapters on one GPU
- **Punica** (Chen et al., 2024) — SGMV kernel for batching distinct adapters in a single GEMM with minimal overhead
- **vLLM / LoRAX / TGI** — production multi-LoRA serving; adapters as lightweight, swappable artifacts over a shared base
- **Task arithmetic** — merging multiple ΔW additively (W₀ + Σⱼ λⱼΔWⱼ) to compose skills, with TIES/DARE to resolve interference

## Common mistake

Merging then quantizing and expecting QLoRA accuracy. The adapter was trained against the *quantized* W₀ (with its quantization error baked in); merging into FP16 W₀ and re-quantizing the sum reintroduces error the adapter can't see, degrading quality. Merge into the same precision the adapter trained against, or serve unmerged.

## See also
- [[lora]] — the underlying ΔW=BA decomposition being merged or served
- [[qlora]] — merging interacts badly with re-quantizing the base
- [[task-arithmetic-task-vectors]] — additive merging of multiple adapters/task vectors
