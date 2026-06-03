# Context Compaction / Summarization

**One-liner:** When the context window fills, periodically replace older turns and verbose tool outputs with an LLM-generated condensed state (salient facts, decisions, open tasks, file paths) so a long-horizon agent can run for hundreds of steps without overflowing or degrading.

## The mechanism

Trigger when token usage crosses a threshold (e.g. usage > 0.8 * window, or every N tool calls). Then:

```
if tokens(history) > threshold:
    keep   = system_prompt + recent_k_turns          # verbatim tail
    old    = history[: -recent_k_turns]              # to be compressed
    summary = LLM.summarize(old, schema=COMPACT_SPEC) # facts, decisions, TODOs, artifacts
    history = system_prompt + [summary] + recent_k_turns
```

`COMPACT_SPEC` is a structured template, not free prose: original goal, key findings/decisions, files touched + paths, tool results worth keeping, current plan, next step. Tool outputs are the prime compaction target — they are large, stale fast, and re-derivable. Distinct from passive eviction (KV-cache compression): compaction rewrites the *token stream* the model sees, so it survives a fresh prefill.

## Where it appears

- **Claude Code / Anthropic agents** — auto-compaction on context-window pressure; "memory tool" persists durable state to a file outside the window so it survives compaction loss.
- **Manus, Cline, OpenAI Codex CLI, Devin-style coding agents** — periodic conversation summarization plus offloading bulky artifacts (logs, file dumps) to disk and re-reading on demand.
- **Long research / deep-research agents** — sub-agents return only compact summaries to a coordinator, combining compaction with context isolation.

## Common mistake

Treating compaction as lossless. It is lossy by construction: any fact not in the summary (and not re-derivable) is gone. The fix is not "summarize harder" but offloading ground truth — scratchpad files, a memory store, the filesystem — so the agent can re-fetch detail instead of relying on a prose recap. Over-aggressive thresholds also break KV-cache prefix reuse, spiking cost.

## See also
- [[kv-cache-aware-agent-design]] — eviction/quantization shrink the cache; compaction rewrites the token stream itself
- [[sub-agent-context-isolation]] — the complementary fan-out strategy: never let context grow rather than compress it later
- [[agentic-memory-architectures]] — external durable memory is what makes lossy compaction safe
