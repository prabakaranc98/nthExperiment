# Model Context Protocol (MCP)

**One-liner:** Open JSON-RPC 2.0 protocol (Anthropic, Nov 2024) standardizing how LLM apps expose and consume tools, resources, and prompts over a client-server link — turning N×M bespoke integrations into N+M, the de facto agent interoperability layer.

## The architecture

Three roles: **Host** (the LLM app, e.g. Claude Desktop, an IDE, an agent) embeds one or more **Clients**; each Client maintains a 1:1 stateful session with a **Server** that wraps a data source or capability. Wire format is JSON-RPC 2.0 over a transport.

Servers expose three primitive types:
- **Tools** — model-controlled functions the LLM may invoke (`tools/list`, `tools/call`); each has a JSON-Schema `inputSchema`. This is server-advertised function calling.
- **Resources** — application-controlled read-only context addressed by URI (`resources/list`, `resources/read`); e.g. `file://`, `postgres://`.
- **Prompts** — user-controlled templated workflows (`prompts/list`, `prompts/get`).

Lifecycle: `initialize` handshake negotiates protocol version + capabilities, then `notifications/initialized`. Servers can also request **sampling** (ask the host's LLM to complete) and **elicitation** (ask the user), inverting the usual control flow.

Transports: **stdio** (local subprocess) or **Streamable HTTP** (remote, replaced the older HTTP+SSE transport in the 2025-03-26 spec). Auth on HTTP servers uses OAuth 2.1.

## Where it appears

- **Claude Desktop / Code, Cursor, Windsurf, VS Code, OpenAI Agents SDK, Gemini CLI** — all adopted MCP as the host-side connector standard through 2025; OpenAI and Google announced support in 2025, ending the single-vendor phase.
- **Agentic tool use** — replaces hand-written function-calling glue; the same Server (GitHub, Slack, Postgres, filesystem) plugs into any compliant host.

## Common mistake

Treating MCP as a remote-API or RPC framework like REST/gRPC. It is an LLM-context protocol: tools/resources/prompts are designed to be surfaced to a model's context window with semantic descriptions, not called by deterministic code. Also: every connected server widens the attack surface — untrusted tool descriptions and returned content are prompt-injection vectors (the lethal trifecta), so a "trusted" client does not imply trusted servers.

## See also
- [[function-tool-calling]] — MCP standardizes the discovery/transport layer around tool calling
- [[agent-security-the-lethal-trifecta]] — connecting external servers exposes injection/exfiltration risk
- [[context-engineering]] — MCP resources/prompts are the plumbing for assembling agent context
