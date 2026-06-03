# Hierarchical / Distributed KV Store (Offload & Transfer)

**One-liner:** Treat the KV cache as a tiered, transferable resource — spill cold blocks down a GPU HBM → host DRAM → NVMe hierarchy and move/share them across nodes over fast interconnects — so reusable context survives eviction and crosses the prefill/decode boundary instead of being recomputed.

## The key insight

KV cache is the dominant inference memory cost (per token, per layer: 2 · n_layers · n_kv_heads · d_head · bytes), but most of it is **cold** — recently unused prefix blocks. Rather than evict-and-recompute, treat it as a paged, addressable store with a tiered backend:

- **Capacity:** HBM (~80GB, ~3TB/s) « DRAM (~TB, ~100s GB/s) « NVMe (~10s TB, ~GB/s). Push cold blocks down; promote on hit.
- **Reuse economics:** a cache hit replaces a full prefill. Worth it iff transfer time < recompute time, i.e. `bytes / BW_link < prefill_FLOPs / FLOPs_rate`. For long shared prefixes (system prompts, RAG docs, multi-turn history), transfer wins by a wide margin — prefill is O(N) compute, fetch is O(N) bytes at high BW.
- **Disaggregation:** in prefill/decode split serving, the prefill node's KV must *move* to the decode node — over NVLink / RDMA / InfiniBand, ideally layer-by-layer overlapped with compute so transfer hides under the next layer.

Effective hit rate, not raw HBM size, sets achievable throughput once a KV pool is shared across requests and nodes.

## Where it appears

- **Mooncake (Moonshot/Kimi, 2024-25)** — KVCache-centric disaggregated architecture; a global "KVCache pool" pools idle DRAM/SSD across the cluster, prefill nodes register prefixes, decode nodes fetch over RDMA.
- **LMCache + vLLM** — open KV layer that offloads/loads blocks to DRAM/NVMe and shares them across vLLM instances; non-prefix reuse and disk-backed long-context.
- **DeepSpeed-Inference / FlexGen** — earlier weight+KV offload to CPU/NVMe for memory-constrained serving.
- **NIXL / KV transfer connectors** — point-to-point KV movement primitives backing prefill/decode disaggregation in production stacks.

## Common mistake

Assuming a remote/CPU KV hit is always cheaper than recompute. If the link is slow (PCIe, congested RDMA) or the prefix is short, transferring the bytes costs more than just re-running prefill — and offloading too aggressively makes you bandwidth-bound, stalling decode. Tier and admit blocks by reuse probability and the transfer-vs-recompute crossover, not by "anything not in HBM goes to disk."

## See also
- [[disaggregated-prefill-decode]] — the serving split that forces KV to transfer between nodes
- [[prefix-caching-radixattention]] — what fills the pool: shared-prefix blocks worth keeping resident
- [[gpu-interconnect-topology]] — NVLink/RDMA bandwidth sets the transfer-vs-recompute crossover
