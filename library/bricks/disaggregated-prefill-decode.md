# Disaggregated Prefill/Decode

**One-liner:** Run prefill and decode on separate GPU pools — each with its own parallelism, batch policy, and replica count — and ship the KV cache between them, so the compute-bound prefill phase and the memory-bandwidth-bound decode phase stop interfering and are each provisioned for their own bottleneck.

## The key insight

The two phases have opposite hardware profiles. Prefill processes the whole prompt in one forward pass: compute-bound, high MFU, sets TTFT (time-to-first-token). Decode emits one token per step over the KV cache: memory-bandwidth-bound, low MFU, sets TPOT/ITL (inter-token latency). Colocating them forces a bad compromise — large prefills stall ongoing decodes (the "decode stall"), and batching for decode throughput hurts prefill latency.

Disaggregation splits them onto distinct instances:

```
[Prefill pool]  prompt → forward pass → KV cache
                          │ transfer KV (NVLink / RDMA / IB)
                          ▼
[Decode pool]   KV cache → autoregressive generation → tokens
```

Each pool independently tunes: TP/PP degree, batch size, and replica ratio (e.g. N prefill : M decode workers, set to balance the two SLOs). You optimize goodput = requests/sec meeting **both** TTFT and TPOT SLOs, instead of raw throughput.

The cost is the KV transfer: per request ≈ `2 · L · n_layers · d_kv · bytes` for L prompt tokens. This must be hidden behind compute (layer-wise / chunked overlap) or it eats the TTFT savings.

## Where it appears

- **DistServe (OSDI 2024)** — formalized phase disaggregation; places prefill/decode on separate GPUs, co-optimizes parallelism + placement for per-phase SLOs, reports large goodput gains under tight latency targets.
- **Mooncake (Moonshot/Kimi)** — KVCache-centric disaggregated architecture; separate prefill and decode clusters plus a global, multi-tier (DRAM/SSD) KV store for cross-request prefix reuse.
- **Splitwise / TetriInfer / DéjàVu** — phase splitting across heterogeneous GPU tiers (cheap GPUs for memory-bound decode, fast GPUs for compute-bound prefill).
- **vLLM / SGLang / NVIDIA Dynamo (2024-2025)** — production P/D disaggregation modes with KV transfer over NVLink/RDMA.

## Common mistake

Assuming disaggregation always wins. It only pays off when the KV transfer is cheap relative to compute (fast interconnect, large enough requests) and load is high enough to keep both pools busy. At low load or on slow links the transfer latency and the extra hop can make it *worse* than colocated **chunked prefill**, which interleaves the two phases on one instance without moving any KV.

## See also
- [[prefill-vs-decode]] — the two-phase asymmetry that disaggregation exploits
- [[chunked-prefill]] — the main colocated alternative; slices prefill to coexist with decode
- [[hierarchical-distributed-kv-store]] — the cross-instance KV layer (Mooncake) that enables transfer + reuse
