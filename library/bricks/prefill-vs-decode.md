# Prefill vs Decode

**One-liner:** The two inference phases — prefill ingests the whole prompt in one compute-bound parallel pass (sets TTFT), decode emits one token per step and is memory-bandwidth-bound by KV-cache reads (sets TPOT) — the root of nearly every serving tradeoff.

## The key distinction

**Prefill** (P prompt tokens): one forward pass over all P tokens at once. Attention is a P×P matmul; FFNs are big GEMMs. Arithmetic intensity is high → **compute-bound** (saturates tensor cores). Produces the first token + fills the KV cache. Cost ≈ O(P) FLOPs-dominant.

**Decode** (one token at a time): each step processes a *single* query token but must read the *entire* KV cache (all previous tokens) and stream all weights from HBM. Tiny matmuls (GEMV-like), low arithmetic intensity → **memory-bandwidth-bound**. Per-step cost dominated by bytes moved, not FLOPs.

Two latency metrics fall straight out:
- **TTFT** (time-to-first-token) ≈ prefill time, scales with prompt length P
- **TPOT / ITL** (time-per-output-token / inter-token latency) ≈ one decode step, roughly constant per token, set by HBM bandwidth and KV size

Rough decode roofline: t_step ≈ (model_bytes + kv_bytes) / HBM_bandwidth. At batch 1 you read all weights to emit one token — terrible MFU; batching amortizes the weight read across many sequences, which is why throughput-serving lives or dies on batch size.

## Where it appears

- **Disaggregated serving** (DistServe, Mooncake, vLLM P/D) — run prefill and decode on *separate* GPU pools because they have opposite bottlenecks; mixing them lets long prefills stall decode steps.
- **Chunked prefill** (Sarathi-Serve) — split a long prefill into chunks and interleave with ongoing decodes so decode latency stays bounded; piggybacks decode tokens onto compute-bound prefill batches to raise MFU.
- **Speculative decoding / EAGLE / Medusa** — attack the decode bottleneck: verify many draft tokens in one bandwidth-bound pass, turning serial decode toward parallel-verify.
- **Prefix caching / RadixAttention** — skip prefill entirely for shared prompt prefixes by reusing cached KV.

## Common mistake

Thinking decode is slow because of FLOPs, so quantizing weights for compute speedup is the fix. Decode is **memory-bandwidth-bound**: the win comes from moving fewer *bytes* (KV-cache compression/quantization, GQA/MLA, bigger batches), not from faster math. Conversely, prefill *is* compute-bound, so FP8 GEMMs and FlashAttention help there but barely move per-token decode latency.

## See also
- [[kv-cache]] — the per-token memory traffic that makes decode bandwidth-bound
- [[disaggregated-prefill-decode]] — splitting the two phases across hardware
- [[chunked-prefill]] — interleaving prefill chunks with decode to balance TTFT and TPOT
