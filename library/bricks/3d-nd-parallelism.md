# 3D / nD Parallelism

**One-liner:** Compose data + tensor + pipeline (+ sequence + expert) parallelism along orthogonal axes of a logical device mesh so a trillion-parameter model fits in memory and trains at high MFU; the organizing framework for every frontier run.

## The key insight

Each parallelism is a different axis of an N-d device grid; total GPUs = product of axis degrees. Classic 3D:

`world_size = DP × TP × PP`   (e.g. 16 × 8 × 12 = 1536 GPUs)

The axes are *orthogonal* — a GPU is identified by its coordinate `(dp, tp, pp)` and each axis has its own communication group. Match collective intensity to interconnect bandwidth:

- **TP** (Megatron) — all-reduce *every layer*, highest traffic → keep inside a node (NVLink/NVSwitch), degree ≤ 8.
- **PP** — point-to-point activations at stage boundaries, cheap → spans nodes; cost is the bubble (~`(p−1)/m`, killed with interleaved/zero-bubble schedules).
- **DP / FSDP / ZeRO** — gradient all-reduce or reduce-scatter once per step, overlappable → outermost axis, scales across the slowest links.
- **SP** shards sequence/activations (folds into TP group); **EP** shards experts of an MoE (its own all-to-all axis). Add axes as needed → "nD".

Mesh sizing: pick TP×PP large enough that `params + optimizer + peak activations` fit per GPU, then fill remaining GPUs with DP for throughput.

## Where it appears

- **Megatron-LM / Megatron-DeepSpeed** — canonical TP×PP×DP (+SP) implementation used for GPT-3, MT-NLG 530B, and most dense frontier pretraining.
- **DeepSeek-V3 / Llama 3 405B / GPT-4-scale runs** — DP(ZeRO/FSDP) + PP across the cluster, TP within a node; MoE models add an EP axis with expert all-to-all.
- **PyTorch `DeviceMesh` / `DTensor`, JAX `Mesh` + `shard_map`** — first-class nD-mesh APIs that let you name axes and shard tensors against them declaratively.

## Common mistake

Treating the axes as a free hyperparameter grid rather than a bandwidth-matched hierarchy. Putting TP across node boundaries strangles you on inter-node all-reduce; making PP too deep starves you on the pipeline bubble; over-sharding with ZeRO-3 across slow links bottlenecks the per-layer all-gather. The art is mapping the *cheapest* collective to the *fastest* link, not maximizing any single degree.

## See also
- [[tensor-parallel]] — the intra-node compute-sharding axis (Megatron f/g all-reduces)
- [[pipeline-parallelism-the-bubble]] — the inter-node stage axis and its idle-time cost
- [[zero]] — the data-parallel memory-sharding leg composed on the outer axis
