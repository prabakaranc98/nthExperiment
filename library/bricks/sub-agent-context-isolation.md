# Sub-Agent / Context Isolation

**One-liner:** Spawn a fresh-context sub-agent to run a bounded subtask (search, read, tool-loop), returning only a distilled result to the parent — keeping the orchestrator's context lean and free of intermediate token bloat.

## The key insight

The parent's context window is the scarce resource. A subtask that requires reading 50 files or 20 tool calls would otherwise dump all of that into the parent, degrading attention and burning budget irreversibly (you cannot un-append tokens). Instead:

```
result = spawn_subagent(
    task        = "find where auth is validated",
    context     = MINIMAL,        # clean window, only the task prompt + tools
    tools       = [Read, Grep],
    budget      = N tokens/steps
)
# subagent burns its own context doing the work,
# returns a SMALL summary (e.g. 3 file paths + 1 line each)
parent_context += result          # parent only pays for the distilled output
```

Information flows *down* as a task spec and *up* as a compressed result; the messy middle (search dead-ends, full file contents, failed tool calls) stays isolated in the child and is discarded when it exits. This is **lossy compression by construction** — the parent never sees the raw evidence, only the conclusion.

## Where it appears

- **Claude Code / Anthropic agent SDK** — the `Task` tool spawns sub-agents with their own context; "do broad search in a subagent, return only the answer" is the canonical pattern.
- **Anthropic multi-agent research system (2024)** — a lead agent fans out parallel sub-agents (each isolated context) and synthesizes; reported large token usage but better coverage on breadth-first tasks.
- **OpenAI Deep Research / Manus / orchestrator-worker frameworks** — workers run isolated tool loops; orchestrator sees only distilled findings.
- **CodeAct / planner-executor agents** — planner stays clean, executors absorb the noisy execution traces.

## Common mistake

Treating sub-agents as free context savings without accounting for **information loss and re-derivation cost**. The parent cannot interrogate what the child saw — if the child's summary is wrong or omits a detail, the parent has no way to recover it short of re-spawning. Sub-agents are great for read-heavy, decomposable, low-coupling tasks; they fail when subtasks share state or need tight back-and-forth (then you pay round-trips and lose the shared scratchpad). It is a complement to compaction, not a substitute: compaction summarizes *in place*, isolation prevents the tokens from ever entering the parent.

## See also
- [[context-compaction-summarization]] — the in-place alternative; isolation prevents bloat, compaction removes it after the fact
- [[multi-agent-orchestration]] — the parent/lead coordinating many isolated workers
- [[context-engineering]] — managing the context window as the core agent design constraint
