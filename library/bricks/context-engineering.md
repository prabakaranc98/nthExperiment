# Context Engineering

**One-liner:** The discipline of deliberately curating exactly what occupies the finite context window each agent step — system prompt, tools, retrieved facts, history, scratchpad — treating the window as a managed budget to maximize signal and minimize distractor tokens.

## The key insight

Every token in the window competes for the same finite attention budget, and quality degrades non-linearly as it fills (the "context rot" / lost-in-the-middle effect). So the goal each step is not "stuff in everything relevant" but "find the smallest set of high-signal tokens that lets the model take the right next action." Frame it as an optimization:

maximize  P(correct next action | context)
subject to  |context| ≤ window  (and effectively ≪ window for good attention)

Four failure modes to engineer against (Drew Breunig / Cognition framing):
- **poisoning** — a hallucination/error enters context and gets referenced repeatedly
- **distraction** — accumulated history drowns out the system prompt and current goal
- **confusion** — irrelevant tools/facts bias the next token
- **clash** — contradictory info across turns (stale vs. fresh state)

Core levers: retrieve just-in-time (pull, don't pre-load), compact/summarize long histories, isolate sub-tasks into fresh windows, persist durable state outside the window (files, memory), and order content so the goal and most-relevant facts sit at the edges, not buried mid-context.

## Where it appears

- **Anthropic "Effective context engineering for AI agents" (2025)** — names the field; "smallest set of high-signal tokens," just-in-time retrieval, compaction at threshold, sub-agent isolation
- **Claude Code / agentic harnesses** — tool results, file reads, and todo lists are compacted/summarized as the window fills; CLAUDE.md as persistent low-cost context
- **Cognition "Don't Build Multi-Agents"** — argues context-sharing failures are why naive multi-agent splits underperform a single well-engineered context
- **RAG and agentic retrieval** — retrieval reframed as context budgeting, not just recall@k
- **KV-cache-aware design** — keeping a stable prefix so cached tokens stay valid (don't reshuffle the system prompt every turn)

## Common mistake

Treating it as "max out the context window." A bigger window is capacity, not a target — more tokens means more distractors and worse attention to the ones that matter. Curate down; relevance density beats volume.

## See also
- [[context-compaction-summarization]] — the primary tool for staying under budget across long sessions
- [[sub-agent-context-isolation]] — give each sub-task a clean window to prevent distraction/clash
- [[kv-cache-aware-agent-design]] — stable-prefix design so context curation doesn't kill cache hits
