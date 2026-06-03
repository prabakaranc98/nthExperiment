# ReAct (Reason + Act)

**One-liner:** Interleave chain-of-thought reasoning with tool actions and environment observations in a loop, so each thought conditions the next action and each observation grounds the next thought — the default scaffold for tool-using LLM agents.

## The loop (pseudocode)

```
context ← [system_prompt, task]
loop:
    thought  ← LLM(context)            # free-text reasoning ("I should search for X")
    action   ← LLM(context)            # tool call: name + args (JSON / function-calling)
    if action == FINISH(answer): return answer
    obs      ← Env.execute(action)     # tool result, search hit, error, page text
    context ← context + [thought, action, obs]
```

The unit is the **(Thought → Action → Observation)** triple, appended to context and repeated until the model emits a terminal answer. Thought is what distinguishes ReAct from a bare act-only tool loop: it makes the *plan explicit in the context window* so subsequent steps are conditioned on stated reasoning, and observations correct hallucinated reasoning before it compounds.

## Where it appears

- **ReAct (Yao et al., ICLR 2023)** — original synergy result: reasoning-only (CoT) hallucinates facts, action-only can't plan; interleaving beats both on HotpotQA, FEVER, ALFWorld, WebShop.
- **Function/tool calling** — every modern agent API (OpenAI tools, Anthropic tool use, MCP servers) is a ReAct loop with structured action serialization instead of parsed text.
- **Frontier coding agents** — Claude Code, Codex, SWE-agent: edit/run/read-output cycles are ReAct where observations are test logs and stack traces.
- **Reflexion / self-reflection** — adds an episodic critique step on top of the base ReAct trajectory.

## Common mistake

Thinking the verbose "Thought:" text is the load-bearing part. The grounding from **observations** is what prevents hallucination — a ReAct loop with weak tools or silently-dropped errors fails no matter how good the reasoning looks. Also: unbounded context growth. Each triple is appended forever, so long trajectories blow the window and degrade; production loops need compaction/summarization or sub-agent isolation, not raw appending.

## See also
- [[function-tool-calling]] — the action-serialization mechanism that implements the Act step
- [[self-reflection-reflexion]] — layers self-critique over ReAct trajectories
- [[context-compaction-summarization]] — fixes the unbounded-context failure mode of long loops
