# KV Cache

**One-liner:** Cache the K,V projections of all past tokens so each autoregressive decode step is O(1) in sequence length instead of O(N); the cache grows linearly with context × layers × KV-heads and is the dominant memory bottleneck of LLM inference.

## The key insight

In causal attention, token t attends to keys/values of all tokens ≤ t. During decode, the K,V vectors of past tokens never change — recomputing them every step is pure waste. Cache them; only project K,V for the *new* token each step.

Per decode step (one new query q_t):
- compute k_t, v_t; append to cache K∈ℝ^{t×d}, V∈ℝ^{t×d}
- attn = softmax(q_t Kᵀ / √d) V  → O(t·d) work, not O(t²·d)

Cache size (bytes), one request:
  **2 · L · H_kv · d_head · S · n_bytes**
(2 = K and V; L = layers; H_kv = KV heads; S = sequence length). E.g. Llama-3-70B at S=8k, fp16, GQA(8 KV heads): ≈ 80·8·128·8192·2·2 ≈ 2.7 GB — *per sequence*. This linear-in-S, linear-in-batch growth, not model weights, caps your context length and throughput.

## Where it appears

- **GQA / MQA** (gqa) — shrink H_kv (group/share KV heads) specifically to cut cache size; the #1 architectural lever (Llama 3, Mistral, Gemini).
- **PagedAttention / vLLM** — page the cache into fixed blocks like virtual memory; kills fragmentation, enables prefix sharing across requests.
- **Prefill vs decode** — prefill fills the cache compute-bound in parallel; decode reads it memory-bound, one token at a time. The split defines serving economics (inference-and-serving).
- **MLA (DeepSeek-V2/V3)** — compress K,V into a low-rank latent, storing the latent instead of full K,V.
- **Quantization** — KV cache in fp8/int4 to roughly halve/quarter footprint.

## Common mistake

Thinking the cache makes attention sublinear. Per step it removes the O(t²) *recompute*, but reading the cache is still O(t) per step and O(S²) total over a generation — and the cache itself grows linearly with context. Long-context decode is bandwidth-bound on cache reads, not FLOP-bound.

## See also
- [[gqa]] — the primary trick to shrink the KV cache by sharing KV heads
- [[flash-attention]] — same memory-bandwidth lens; complementary at prefill/decode
- [[quantization]] — fp8/int4 KV cache to fit longer context and bigger batches
