# PagedAttention

**One-liner:** Store the KV cache in fixed-size non-contiguous physical blocks indexed by a per-sequence block table (OS-style virtual paging), eliminating fragmentation and enabling near-zero-copy prefix sharing and copy-on-write forks (vLLM).

## The key insight

Naive serving allocates one contiguous buffer per request sized for `max_seq_len`, wasting most of it (internal fragmentation) and stranding free gaps (external fragmentation). PagedAttention instead partitions KV into **blocks of B tokens** (e.g. B=16). Logical token i lives in:

```
block_idx     = i // B
block_offset  = i % B
phys_block     = block_table[seq_id][block_idx]   # logical -> physical map
addr           = phys_block * B + block_offset
```

The attention kernel gathers K,V through this indirection — physical blocks need not be contiguous. Blocks are allocated lazily as the sequence grows, so waste is bounded by < 1 block per sequence (vs. ~60-80% in naive serving).

## Where it appears

- **vLLM (Kwon et al., SOSP 2023)** — the origin; pairs PagedAttention with continuous batching to push serving throughput 2-4x over prior systems by raising the usable batch size.
- **Prefix sharing / RadixAttention** — a shared system prompt's blocks are referenced by many sequences via their block tables; one physical copy, many readers.
- **Copy-on-write forking** — parallel sampling (best-of-n, beam search) shares the prompt's blocks; a block is duplicated only on first write divergence, cutting prompt-KV memory.
- **TensorRT-LLM, SGLang, TGI** — all adopt paged KV block managers as the default memory layout.

## Common mistake

Thinking the block table indirection is "free." The gather adds a layer of pointer-chasing to the attention kernel, so PagedAttention needs a custom fused kernel (and CUDA-graph-friendly layouts) to avoid a per-token latency tax. It trades a small kernel-complexity/overhead cost for a large gain in memory utilization and batch size — the win is throughput via packing, not lower single-request latency.

## See also
- [[kv-cache]] — the structure PagedAttention is paging; the memory it manages
- [[prefix-caching-radixattention]] — builds shared-prefix reuse on top of paged blocks
- [[continuous-batching]] — the co-designed scheduler that exploits the freed memory
