# KV-Cache-Aware Agent Design

**One-liner:** Engineer agent prompts as a stable, append-only token stream so the longest possible prefix hits the provider's prompt cache, turning O(N²) recompute over a multi-turn agent loop into O(N) and slashing TTFT and per-call cost ~10x.

## The key insight

Prompt caches (Anthropic, OpenAI, Google, vLLM/SGLang radix trees) key on the **longest common token prefix** of consecutive requests. The KV-cache for that prefix is reused; only the new suffix tokens are prefilled. Caching is **prefix-exact**: a single differing byte anywhere invalidates everything from that byte onward.

Agent cost over a T-step ReAct loop with growing context:
- No cache: total prefill ≈ Σ_{t=1..T} (P + t·Δ) → **O(T²·Δ)** tokens recomputed
- Prefix cache: each step recomputes only its new suffix ≈ Σ Δ → **O(T·Δ)**

Design rules that follow directly:
1. **Order by stability:** system prompt → tool definitions → few-shot exemplars → retrieved docs → conversation → newest turn. Volatile content goes last.
2. **Append-only history:** never edit, reorder, or re-summarize earlier turns mid-trajectory; that moves the divergence point earlier and dumps the whole tail.
3. **No nondeterminism in the prefix:** strip timestamps, request UUIDs, randomized JSON key order, and per-call seeds out of the cached region.
4. **Tool-result placement:** keep stable tool schemas in the prefix; let only the variable tool *outputs* land in the suffix.
5. **Mark explicit cache breakpoints** (Anthropic `cache_control`; OpenAI/Google auto-cache after a min prefix length, ~1024 tokens).

## Where it appears

- **Anthropic prompt caching** — explicit `cache_control` breakpoints, 5-min/1-hr TTL; cache writes cost ~1.25x, cache reads ~0.1x base input price. The pricing asymmetry is what makes prefix discipline a first-class design constraint.
- **vLLM / SGLang RadixAttention** — automatic prefix sharing across requests via a radix tree over KV blocks; concurrent agents sharing a system prompt all hit the same cached blocks.
- **Manus / production agent harnesses** — publicly cite "maximize KV-cache hit rate" as their #1 latency/cost lever, including append-only context and stable tool masking instead of dynamic tool lists.
- **Claude Code / coding agents** — stable system+tools prefix kept identical across the whole session so each turn only prefills the new file diffs and user message.

## Common mistake

Mutating the prefix to "save tokens" — e.g., re-summarizing or pruning old turns, sorting tool definitions alphabetically, or stamping the current time into the system prompt. Each of these moves the first-differing token earlier, evicting the cached suffix and forcing a full recompute. The token "savings" are dwarfed by losing the cache; an append-heavy but stable prompt is almost always cheaper than a compact but volatile one.

## See also
- [[prefix-caching-radixattention]] — the serving-layer mechanism that makes prefix reuse automatic
- [[kv-cache]] — the underlying state that is being shared and reused
- [[context-engineering]] — caching is a hard constraint on how you order and edit agent context
