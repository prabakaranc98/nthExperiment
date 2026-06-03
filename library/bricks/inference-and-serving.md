# Inference & Serving

**One-liner:** The decode-time stack — compute-bound *prefill* followed by memory-bound *decode*, scheduled via continuous batching over a paged KV cache with prefix reuse — to maximize tokens/sec/$ under p50/p99 latency SLOs.

## The two phases

- **Prefill:** process the whole prompt in one parallel forward pass. Compute-bound (big matmuls, high arithmetic intensity). Sets **TTFT** (time to first token).
- **Decode:** generate one token at a time, each reading the full KV cache. Memory-bandwidth-bound (one query row × all cached keys/values). Sets **TPOT/ITL** (time per output token / inter-token latency).

Per-step decode cost ≈ read of all weights + KV cache from HBM ⇒ throughput is gated by bandwidth, not FLOPs. Hence batching is the lever: amortize the weight read across many sequences.

## Continuous (in-flight) batching

Naive static batching stalls: the batch waits for the slowest sequence to finish. Continuous batching schedules at the **iteration** level — evict finished sequences and admit new ones every decode step, keeping the GPU saturated. Origin: Orca (2022). Standard in vLLM, TGI, TensorRT-LLM, SGLang.

## PagedAttention & prefix caching

KV cache is the memory bottleneck: size = `2 · layers · heads_kv · d_head · seq_len · batch · bytes`. Contiguous allocation wastes 60–80% to fragmentation/over-reservation.

**PagedAttention** (vLLM, 2023): store KV in fixed-size **blocks** via a block table (OS-style paging), so memory grows on demand and is shared. Enables **prefix caching** — identical prompt prefixes (system prompts, few-shot) share KV blocks copy-on-write, skipping their prefill entirely. SGLang's RadixTree generalizes this to a trie of cached prefixes.

## Where it appears

- **vLLM / SGLang / TensorRT-LLM** — PagedAttention + continuous batching + prefix/RadixCache are the default serving substrate (2024–2026).
- **Disaggregated prefill/decode** (DistServe, Mooncake, 2024) — run compute-bound prefill and bandwidth-bound decode on *separate* GPU pools to hit TTFT and TPOT SLOs independently.
- **Chunked prefill** — interleave prompt chunks with decode steps so long prompts don't head-of-line-block ongoing generations.

## Common mistake

Optimizing average throughput while ignoring the latency SLO. Bigger batches raise tokens/sec but inflate p99 TPOT; prefill and decode have *opposite* hardware bottlenecks, so a single knob can't satisfy both. Tune for the SLO (TTFT vs TPOT), not for raw tokens/sec.

## See also
- [[kv-cache]] — the object PagedAttention pages and prefix caching shares
- [[speculative-decoding]] — turns memory-bound decode into verified parallel batches
- [[roofline]] — why prefill is compute-bound and decode is bandwidth-bound
