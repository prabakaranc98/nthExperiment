# KV-Cache Compression / Eviction

**One-liner:** Bound KV-cache memory by keeping only a subset of token K,V (attention sinks + recent window, or top-scoring tokens) and dropping/merging the rest, trading a small accuracy hit for sublinear-in-context memory — distinct from KV *quantization*, which shrinks bits per kept token rather than the token count.

## The key insight

Attention is sparse: most decode queries put nearly all their mass on a few "heavy hitter" tokens plus the recent window, while early tokens act as attention sinks (huge mass on position 0 regardless of content). So you can evict the rest. The eviction policy chooses *which* tokens to keep within a budget B:

- **StreamingLLM:** keep first k sink tokens + a sliding window of the most recent W. Cache size fixed at k+W; enables infinite-length streaming, but truly drops middle context (no recall of evicted tokens).
- **H2O (Heavy-Hitter Oracle):** score token j by accumulated attention it has received, A_j = Σ_t softmax_t[j]; greedily keep top-B by A_j plus the recent window. Online, per-decode-step eviction.
- **SnapKV:** at the *end of prefill*, use the last few "observation" query rows to vote which prompt KV to keep (pool attention scores, take top-B per head), then decode normally. Compress once, no per-step eviction.

Net effect: cache ≈ O(B) instead of O(S). Eviction is permanent (info loss); compression here means fewer tokens, orthogonal to per-token bit-width.

## Where it appears

- **StreamingLLM (Xiao et al. 2023)** — sink+window; the result that "first tokens are sinks" became standard serving folklore.
- **H2O (Zhang et al. 2023)** — accumulated-attention eviction; canonical heavy-hitter baseline.
- **SnapKV / PyramidKV (2024)** — prefill-time prompt compression; PyramidKV varies budget B by layer (more in lower layers). Used to fit long agentic/RAG prompts.
- **vLLM / SGLang serving** — eviction policies layered on PagedAttention block management for long-context and multi-turn agents.

## Common mistake

Treating eviction as lossless like FlashAttention or quantization. Evicted tokens are *gone* — accuracy collapses on tasks needing the dropped middle (needle-in-haystack, exact retrieval, long-horizon agents). The attention-score importance estimate is also a leaky proxy: a token unimportant for past queries can be critical for a future one. Budget and policy are task-dependent, not free.

## See also
- [[kv-cache]] — the linear-in-context memory bottleneck this attacks
- [[attention-sinks]] — why keeping position-0 tokens is non-negotiable
- [[kv-cache-quantization]] — orthogonal axis: fewer bits per kept token, not fewer tokens
