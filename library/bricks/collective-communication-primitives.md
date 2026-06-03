# Collective Communication Primitives

**One-liner:** The fixed set of GPU group-communication operations (all-reduce, reduce-scatter, all-gather, all-to-all, broadcast) whose ring/tree algorithms set the bandwidth-latency cost of every parallelism scheme — all-reduce = reduce-scatter + all-gather.

## The primitives and their costs

For P ranks, message size N bytes, link bandwidth B, per-hop latency α:

- **all-reduce**: every rank ends with the elementwise sum (or max/...) of all inputs. Ring algorithm = reduce-scatter then all-gather. Bus bandwidth cost ≈ 2(P−1)/P · N/B → **~2N/B** as P grows (each byte traverses the wire ~2x, independent of P). Latency ~2(P−1)α.
- **reduce-scatter**: sum across ranks, but each rank keeps only its 1/P shard. Cost ≈ (P−1)/P · N/B ≈ **N/B**.
- **all-gather**: each rank holds a 1/P shard; all end with the full concatenation. Cost ≈ **N/B** (mirror of reduce-scatter).
- **all-to-all**: rank i sends a distinct chunk to every rank j (transpose). Cost ≈ (P−1)/P · N/B but with P² messages → latency-bound at scale.
- **broadcast / reduce**: root → all (or all → root). One-to-many, tree algorithm, ~log P latency.

Key identity: **all-reduce(x) = all-gather(reduce-scatter(x))**. ZeRO/FSDP exploit this to never materialize a full all-reduce.

## Where it appears

- **Data parallelism / DDP** — gradient all-reduce every step; the dominant comm cost, overlapped with backward.
- **ZeRO / FSDP** — replaces all-reduce with reduce-scatter (gradients) + all-gather (params/optimizer shards) to cut memory; same 2N/B traffic, different schedule.
- **Tensor parallelism (Megatron)** — all-reduce after each attention/MLP block (or all-gather + reduce-scatter with sequence parallelism); latency-critical, kept intra-node on NVLink.
- **Expert parallelism (MoE)** — two all-to-all per layer to dispatch/combine tokens across experts; the MoE comm bottleneck.
- **NCCL/RCCL** — the implementations; auto-select ring vs tree, and on Hopper/Blackwell use NVLS (in-switch SHARP reduction) to halve all-reduce traffic.

## Common mistake

Thinking all-reduce cost grows with P. The ring all-reduce is bandwidth-optimal at **~2N/B regardless of P** — only latency scales (linearly for ring, log for tree). The real scaling pain is latency and topology (cross-node IB vs intra-node NVLink), not per-GPU bandwidth, which is why TP stays inside a node and DP crosses nodes.

## See also
- [[zero]] — decomposes all-reduce into reduce-scatter + all-gather to shard state
- [[gpu-interconnect-topology]] — NVLink vs InfiniBand bandwidth sets which collective goes where
- [[computation-communication-overlap]] — hiding collective latency behind compute
