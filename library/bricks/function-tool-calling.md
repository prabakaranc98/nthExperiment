# Function / Tool Calling

**One-liner:** The LLM emits a structured (usually JSON) call to a developer-declared function whose execution result is fed back into context as a new message — the universal interface between a language model and the outside world (APIs, code, retrieval, other agents).

## The mechanism

The loop, made explicit:

1. **Declare** tools as JSON Schema: `[{name, description, parameters: {<JSON Schema>}}]`. The schema is rendered into the prompt (or constrained via the decoder).
2. **Model emits** a structured call: `{"name": "get_weather", "arguments": {"city": "Paris"}}` — usually as a distinct content block / message role (`tool_use` in Anthropic, `tool_calls` in OpenAI), not free text.
3. **Harness executes** the function out-of-band (the model never runs code itself) and appends the result as a `tool_result` / `tool` message keyed by a `tool_call_id`.
4. **Re-invoke** the model with the appended result. Repeat until the model emits a final text answer with no tool call.

Training: tool use is taught via SFT on traces + RL (RLVR with execution feedback). The decision of *whether* and *which* to call is learned; the *syntax* is often hard-guaranteed by constrained decoding against the schema grammar.

## Where it appears

- **OpenAI / Anthropic / Gemini APIs** — native `tools` parameter; parallel tool calls (multiple calls in one turn), forced/auto/none tool choice.
- **Model Context Protocol (MCP)** — standardizes tool *discovery and transport* so any client can expose tools to any model; tool calling is the wire format underneath.
- **ReAct / agents** — tool calling is the "Act" step interleaved with reasoning; the backbone of computer-use and browser agents.
- **RAG** — retrieval exposed as a `search(query)` tool the model decides to invoke (agentic/iterative retrieval) rather than a fixed pre-pended context.
- **Constrained decoding** — JSON-mode / grammar masking guarantees the emitted call parses against the schema.

## Common mistake

Treating the model as the executor and trusting its output as ground truth. The model only *proposes* a call — your harness executes it, and the model can hallucinate arguments, call nonexistent tools, or be steered into malicious calls via injected tool results (prompt injection). Always validate arguments against the schema and treat returned tool content as untrusted input, not instructions.

## See also
- [[model-context-protocol]] — the standard for exposing tools to models
- [[react]] — interleaves tool calls with explicit reasoning
- [[constrained-structured-decoding]] — guarantees the emitted call is valid JSON/schema
