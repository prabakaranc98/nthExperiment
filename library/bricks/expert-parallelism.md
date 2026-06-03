# Expert Parallelism (MoE)

**One-liner:** Shard MoE experts across GPUs and route tokens to them via all-to-all dispatch/combine collectives, scaling total parameters without scaling per-token FLOPs — at frontier scale the all-to-all becomes the dominant cost.

## The mechanism

With E experts and an EP group of size P, each device holds E/P experts. A layer runs in four phases:

  1. route:     g = softmax(x·Wᵣ); pick top-k experts per token (local)
  2. dispatch:  all-to-all  — send each token to the device(s) owning its experts
  3. compute:   each device runs its local experts on the tokens it received
  4. combine:   all-to-all  — return expert outputs to each token's origin device, weight-sum

So **2 all-to-alls per MoE layer per direction** (4 total with backward). Volume per all-to-all ≈ (tokens × k × d_model) bytes, scattered across P devices — pattern is irregular and load-dependent. Distinct from tensor parallelism (which shards each expert's matrices and uses all-reduce); EP keeps each expert intact on one device and shards across the expert dimension. The two compose: TP/ETP within an expert, EP across experts.

## Where it appears

- **GShard / Switch Transformer** — introduced expert sharding + the dispatch/combine all-to-all; capacity factor drops overflow tokens to keep buffers static-shaped.
- **DeepSeek-V3 (671B, 37B active)** — large EP degree (e.g. 64+) with **node-limited routing** (cap experts-per-token to ≤4 nodes) to bound cross-node all-to-all; DualPipe overlaps comm with compute.
- **DeepEP / Megatron-Core / Tutel** — production all-to-all kernels over NVLink + RDMA, hierarchical (intra- then inter-node) dispatch.
- **Disaggregated MoE inference (2024–25)** — prefill vs decode use different EP layouts; high-EP decode trades all-to-all latency for per-GPU memory headroom.

## Common mistake

Treating EP as a memory-saving or free scaling axis. All E experts must reside in aggregate VRAM, and EP adds *two synchronizing all-to-alls per layer* on the critical path — latency-bound and sensitive to load imbalance (a hot expert stalls the whole collective). The win is FLOP efficiency at fixed quality; the bottleneck is interconnect bandwidth and routing balance, not arithmetic.

## See also
- [[moe-routing]] — EP is the distributed-systems realization of the sparse top-k router
- [[expert-load-balancing]] — imbalance turns the all-to-all into a straggler-bound stall
- [[tensor-parallel]] — the orthogonal sharding axis; composes with EP, but uses all-reduce not all-to-all
