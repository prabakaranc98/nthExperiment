# Prefix Caching / RadixAttention

**One-liner:** Reuse already-computed KV-cache blocks for shared prompt prefixes (system prompts, few-shot exemplars, conversation history, RAG docs) keyed by a hash or radix tree with LRU eviction — skipping their prefill entirely for huge TTFT and cost wins on chat, RAG, and agents.

## The key insight

KV for token i depends only on tokens [0..i] (causal). So two requests sharing a prefix produce *identical* KV for that prefix — compute it once, reuse it. The shared-prefix property is exactly what a **trie/radix tree over token sequences** captures.

```
RadixTree node = (token edge label, KV blocks, ref/last-access)
on request tokens T:
    matched, node = tree.longest_prefix_match(T)   # share KV for matched
    prefill only T[len(matched):]                   # the unique suffix
    insert new KV blocks under node                 # extend the tree
evict: LRU over leaf KV blocks under memory pressure
```

Match granularity is the page/block (e.g. 16 tokens, à la PagedAttention) so cached and new KV interleave in the same paged pool. Compute saved ≈ shared_prefix_len / total_prefill_len; only prefill FLOPs/latency drop — decode is unchanged.

## Where it appears

- **SGLang (RadixAttention, 2024)** — the canonical implementation: radix tree over request token sequences with LRU eviction, automatic cross-request and multi-turn KV reuse.
- **vLLM Automatic Prefix Caching** — block-hash keyed cache layered on PagedAttention; identical prefix blocks dedup across requests.
- **Anthropic / OpenAI / Gemini prompt caching** — explicit cache breakpoints; cached input tokens billed ~0.1x and skip recompute (with a TTL, e.g. 5 min).
- **Agents / RAG / chat** — stable system prompt + long history + tool defs reused every turn; KV-cache-aware agent design keeps prefixes append-only to maximize hit rate.

## Common mistake

Assuming any text overlap helps. The match must be an *exact token-prefix from position 0* — a single differing early token (a timestamp, reordered tool list, or non-determinism in tokenization) breaks the chain and invalidates everything after it. Put volatile content (current time, user query) at the *end*, stable content at the front.

## See also
- [[kv-cache]] — prefix caching is cross-request reuse of exactly these tensors
- [[pagedattention]] — block/page allocation makes prefix sharing practical
- [[kv-cache-aware-agent-design]] — structuring prompts append-only to maximize cache hits
