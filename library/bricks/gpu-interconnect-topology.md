# GPU Interconnect Topology (NVLink / NVSwitch / InfiniBand)

**One-liner:** A bandwidth hierarchy — NVLink/NVSwitch give ~TB/s all-to-all *within* a node while InfiniBand/Ethernet give ~tens-of-GB/s *across* nodes — so parallelism axes are mapped to the topology (TP intra-node, DP/PP/EP inter-node) to keep the chattiest collectives on the fattest links.

## The key insight

Communication cost = volume / link_bandwidth. Tensor parallelism issues an all-reduce *per layer* (two per transformer block, forward+backward), so its traffic is enormous and latency-sensitive — it must live on NVLink. Pipeline/data parallelism communicate far less per step (activations at boundaries; gradient all-reduce once per step, hideable), so they tolerate slow InfiniBand.

Approximate 2024-2026 bandwidths (per-GPU, unidirectional-ish):
- NVLink 4 (H100): ~450 GB/s; NVLink 5 (GB200): ~900 GB/s, full all-to-all via NVSwitch.
- NVSwitch fabric: non-blocking within an 8-GPU node (or 72-GPU NVL72 rack domain on GB200).
- InfiniBand NDR: 400 Gb/s = 50 GB/s per NIC; one NIC per GPU typical (rail-optimized).
- => intra:inter bandwidth ratio is roughly 10-20x. Ring all-reduce time scales as `2(n-1)/n · V / BW`; the slowest link dominates.

Rule of thumb: `TP degree <= GPUs_per_NVLink_domain` (8, or 72 on NVL72). Beyond that, TP crosses the slow fabric and MFU collapses.

## Where it appears

- Megatron-LM / 3D parallelism — explicitly places TP inside the node, PP+DP across nodes; the canonical bandwidth-to-axis mapping.
- DeepSeek-V3 / Mixtral MoE — expert parallelism dispatches tokens all-to-all; NVL72's 72-GPU NVLink domain lets EP scale before hitting IB.
- DGX / GB200 NVL72 — NVLink domain expanded to a whole rack so TP/EP groups no longer bottleneck on InfiniBand.
- Rail-optimized IB fabrics + SHARP in-network reduction — accelerate the inter-node gradient all-reduce for DP.

## Common mistake

Setting TP degree larger than the NVLink domain (e.g., TP=16 on 8-GPU nodes), forcing per-layer all-reduces over InfiniBand. The per-layer collective traffic then saturates the slow fabric and tanks throughput — TP must stay inside the fast intra-node domain.

## See also
- [[3d-nd-parallelism]] — the axes (TP/PP/DP/EP) being mapped onto this topology
- [[collective-communication-primitives]] — all-reduce/all-to-all whose cost this bandwidth hierarchy bounds
- [[computation-communication-overlap]] — how PP/DP hide slow inter-node links behind compute
