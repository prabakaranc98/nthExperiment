# Multi-Agent Orchestration (Orchestrator-Worker)

**One-liner:** A lead/orchestrator agent decomposes a task, spawns specialized worker subagents that each run in their own context window (parallel or sequential), and synthesizes their compressed returns — the dominant pattern for scaling work beyond a single context window and for embarrassingly-parallel search/read.

## The key insight

The win is **context isolation + parallelism**, not "more agents = smarter." Each worker burns its own tokens on tool-call clutter (long search results, file dumps) and returns only a distilled artifact to the orchestrator, keeping the lead's context clean for high-level planning and final synthesis.

```
orchestrator(task):
    plan = decompose(task)                      # boundaries, success criteria per subtask
    results = parallel_map(run_worker, plan)    # fan-out; each worker = fresh context
    return synthesize(results)                  # fan-in; orchestrator only sees summaries

run_worker(subtask):                            # isolated context window
    loop: act / call_tools / observe until done
    return compress(findings)                   # NOT the raw transcript
```

Cost scales with the number of agents x tokens each: an orchestrator-worker run can use ~15x the tokens of a single chat. It only pays off when subtask value is high and tasks parallelize. Workers are stateless re-spawns; durable state lives in the orchestrator or an external store/filesystem.

## Where it appears

- Anthropic multi-agent research system (2024-25) — lead Claude spawns parallel subagents for breadth-first research; reports ~90% improvement over single-agent on internal eval, attributes most gains to parallel token spend.
- Claude Code / coding agents — main agent dispatches read-only "explore" subagents and `Task`-tool workers; orchestrator keeps a thin plan while workers do the dirty searching.
- OpenAI Swarm / LangGraph / CrewAI / AutoGen — framework primitives for handoffs, supervisor graphs, and role-typed worker pools.

## Common mistake

Treating it as a chat among peers and passing full transcripts around. That blows up context, multiplies cost, and causes workers to duplicate or conflict. The orchestrator's job is crisp subtask boundaries + objective synthesis; workers must return *compressed* results, not their raw scratchpad. Also: parallelizing inherently sequential, dependency-laden tasks just adds coordination overhead and error compounding.

## See also
- [[sub-agent-context-isolation]] — the core mechanism that makes fan-out pay off
- [[context-engineering]] — managing the orchestrator's window budget under fan-in
- [[agent-security-the-lethal-trifecta]] — more agents/tools widens the attack surface
