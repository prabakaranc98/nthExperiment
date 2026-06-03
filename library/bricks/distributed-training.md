# Distributed Training (Parallelism Overview)

**One-liner:** Frontier training composes orthogonal parallelism axes — data, tensor, pipeline, sequence/context, and expert — into a 3D/4D mesh, each axis trading a different communication primitive against memory and compute to keep thousands of GPUs busy.

## The axes (what gets split, what gets communicated)

- **Data parallelism (DP):** replicate model, split the batch. Sync gradients via **all-reduce** each step. Comm ∝ #params, independent of batch. ZeRO/FSDP shard the optimizer/grads/params across DP ranks → trades all-reduce for **reduce-scatter + all-gather**.
- **Tensor parallelism (TP):** split individual matmuls (and attention heads) across GPUs. **all-reduce twice per transformer block** (fwd+bwd). Latency-bound, so keep TP **within one NVLink node** (≤8 GPUs).
- **Pipeline parallelism (PP):** split *layers* across GPUs; microbatches flow stage→stage via **point-to-point send/recv**. Cheap comm but introduces a **bubble** ≈ (p−1)/(m+p−1) for p stages, m microbatches → use many microbatches + interleaved/1F1B schedules.
- **Sequence/context parallelism (SP/CP):** split the *sequence* dimension. SP shaves activation memory in LN/dropout; **Ring Attention / context parallel** shards the KV across GPUs for million-token contexts via overlapped P2P ring exchange.
- **Expert parallelism (EP):** place MoE experts on different GPUs; tokens routed via **all-to-all** (the dominant MoE comm cost).

## The combination (3D / 4D / 5D)

Total GPUs **G = DP × TP × PP** (× CP × EP). Megatron-style ordering: **TP innermost (NVLink) → PP across nodes → DP outermost (sharded grads)**. Per-GPU params ≈ N / (TP·PP·EP); per-GPU optimizer state ≈ that / DP-shard.

Rough comm cost per step: DP ~all-reduce(params), TP ~all-reduce(activations)·layers, PP ~send(activations)·stages, EP ~all-to-all(tokens·d).

## Where it appears

- **Megatron-LM / Megatron-Core** — defines TP+PP+SP; the reference 3D+ implementation.
- **DeepSpeed ZeRO / PyTorch FSDP(2)** — sharded data parallelism; ZeRO-3 ≈ FSDP. Backbone of most open training.
- **GPT-4 / Llama 3 405B / DeepSeek-V3** — Llama 3 used 4D (TP·PP·CP·DP) on 16k H100s; DeepSeek-V3 added large-scale EP with all-to-all overlap.
- **Ring Attention / Gemini long-context** — context parallel for ≥1M tokens.

## Common mistake

Treating the axes as interchangeable knobs. They are not — each maps to a *different collective* with different bandwidth/latency behavior. Putting TP across slow inter-node links (instead of NVLink) tanks throughput; cranking PP without enough microbatches wastes compute in the bubble. You pick the split to match the *network topology*, not just to fit memory.

## See also
- [[fsdp]] — sharded data parallelism, the most common single axis
- [[zero]] — the memory-sharding scheme underlying FSDP/ZeRO-3
- [[tensor-parallel]] — the intra-block matmul split detailed
