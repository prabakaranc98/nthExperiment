# Agentic Memory Architectures

**One-liner:** An OS-like memory hierarchy for LLM agents — a small in-context working set (scratchpad) backed by external long-term stores (vector/graph/file/SQL) that the agent reads and writes via tool calls to persist state across steps and sessions beyond the finite context window.

## The architecture

Treat the context window as RAM and external stores as disk (MemGPT's "LLM as OS" analogy). Each turn, the agent's effective state is:

state = system_prompt ⊕ working_memory ⊕ retrieve(query, long_term_store)

- **Working memory (scratchpad):** in-context, mutable, token-bounded. Holds recent turns, current plan, key facts. Self-edited or summarized when it nears the limit.
- **Long-term store:** out-of-context. Three common backends:
  - **Vector:** chunk → embed → top-k ANN retrieval by cosine similarity (semantic recall, "memory streams").
  - **Graph:** entities/relations as nodes/edges; multi-hop traversal (e.g., Zep/Graphiti temporal graphs, GraphRAG).
  - **File/structured:** named files, KV, or SQL the agent edits explicitly (Claude/MCP "memory tool").

Control loop (MemGPT-style): when the prompt fills, the LLM emits a function call to **evict** (page out a summary to long-term) and **page in** relevant memories on demand — self-managed paging via tool calls, not a fixed window.

Generative Agents memory stream adds a **retrieval score** = α·relevance(embed sim) + β·recency(exp decay) + γ·importance(LLM-rated 1–10), then periodically **reflects** to synthesize higher-level memories from raw observations.

## Where it appears

- **MemGPT / Letta (2023–24)** — virtual context management; LLM self-pages between main context and external recall storage via function calls.
- **Generative Agents (Park et al., 2023)** — memory stream + recency/importance/relevance scoring + reflection trees.
- **Zep / Graphiti, GraphRAG** — temporal knowledge graphs as agent long-term memory; bi-temporal facts, multi-hop queries.
- **MCP memory tool / Claude Agents, Cursor & Devin** — file-based persistent memory edited across sessions; project notes that survive context resets.

## Common mistake

Treating retrieval as a dump: stuffing top-k vector hits into context every turn. This bloats the prompt, buries the signal (lost-in-the-middle), and conflates relevance with usefulness. Good memory is *write-and-curate* — summarize, deduplicate, decay, and reflect — not just append-and-retrieve. Also: vector recall alone has no notion of recency or contradiction; stale facts get retrieved as if current.

## See also
- [[context-engineering]] — managing the finite window memory pages into and out of
- [[rag]] — the retrieval substrate behind vector-backed long-term memory
- [[sub-agent-context-isolation]] — partitioning state across agents instead of one shared store
