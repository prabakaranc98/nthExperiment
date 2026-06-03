# Agentic / Iterative Retrieval

**One-liner:** Turn RAG into a control loop — the model plans, issues multiple queries, reads results, decides whether evidence is sufficient, and reformulates or stops; retrieval becomes a multi-step search policy instead of a single fixed fetch.

## The loop

```
state = query; context = []
while not done and step < budget:
    action = policy(state, context)        # SEARCH(q) | READ(url) | REFLECT | ANSWER
    if action == ANSWER: break
    obs = tool(action)                      # retrieved docs / page text
    context = compact(context + obs)        # summarize to fit window
    done = sufficiency_check(query, context)
answer = generate(query, context)
```

Decisions are made by the LLM via tool-calling / ReAct-style traces. The policy is either **prompted** (zero-shot reasoning), **SFT-distilled** from search trajectories, or **RL-trained** with an outcome reward (final-answer correctness, optionally minus a per-query cost penalty).

## Where it appears

- **Self-RAG (Asai et al., 2024)** — model emits *reflection tokens* (`Retrieve?`, `IsRelevant`, `IsSupported`) to decide on-demand whether and what to retrieve, and to critique its own output.
- **Search-R1 / R1-Searcher / DeepRetrieval (2025)** — RL (GRPO/PPO) over interleaved `<think>…</think><search>…</search>` traces; reward = answer correctness, teaching when to search and how to reformulate.
- **OpenAI Deep Research, Gemini Deep Research, Perplexity (2025)** — long-horizon agents fanning out dozens-to-hundreds of queries, reading pages, and synthesizing cited reports.
- **IRCoT / FLARE** — interleave chain-of-thought with retrieval, triggering a new query when next-token confidence drops.

## Common mistake

Conflating "more retrieval steps" with "better." Each extra hop adds noisy/distracting context that can *lower* accuracy (lost-in-the-middle, context rot) and burns latency/tokens. The hard part is the **stopping policy and query quality**, not the number of loops — over-retrieval is a real failure mode, and an under-trained sufficiency check leads to runaway loops or premature answers.

## See also
- [[rag]] — the single-fetch baseline this generalizes into a search loop
- [[react]] — the reason/act/observe scaffold that implements the loop
- [[context-compaction-summarization]] — required to keep accumulated evidence within the window
