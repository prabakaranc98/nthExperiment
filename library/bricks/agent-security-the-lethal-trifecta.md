# Agent Security & the Lethal Trifecta

**One-liner:** A tool-using agent is exploitable for data exfiltration precisely when three capabilities co-occur — access to private data, exposure to untrusted content, and the ability to externally communicate — so defenses break the trifecta via isolation, provenance tracking, and capability/egress limits (AgentDojo, CaMeL).

## The key insight

Simon Willison's **lethal trifecta**: an attack needs all three legs simultaneously.
1. **Private data** — the agent can read secrets (emails, files, tokens).
2. **Untrusted content** — attacker-controlled text enters the context (a web page, an email, a tool output) → prompt injection.
3. **Exfiltration channel** — the agent can send data out (HTTP fetch, email, markdown image `![](url?leak=...)`, tool call to an external endpoint).

Remove any one leg → exfiltration is impossible. The danger is that LLMs cannot reliably distinguish trusted instructions from injected ones in a flat context window; alignment/detection is **not** a guarantee. So treat it as an access-control problem, not a prompt problem.

Threat model: the **data flow**, not the prompt. Injected text → control-flow hijack → exfil. Formally, deny any path `tainted_input → privileged_action` where the action has external reach and the data is sensitive (an information-flow / taint-tracking property).

## Where it appears

- **AgentDojo (NeurIPS 2024)** — dynamic benchmark of 97 realistic tasks + 629 injection attacks; measures utility under attack and attack success rate, exposing that no defense is robust by construction.
- **CaMeL (Debenedetti et al., 2025)** — dual-LLM: a privileged planner emits a typed control/data-flow plan from trusted input only; a quarantined LLM parses untrusted content; a custom interpreter enforces capabilities/provenance so tainted data never reaches sensitive sinks. Provable, not heuristic.
- **Dual-LLM / "spotlighting" patterns** — quarantined model handles untrusted text, returns only structured/symbolic results; the action-taking model never sees raw injected tokens.
- **MCP / browser & computer-use agents** — primary real-world attack surface; egress allowlists, human-in-the-loop for irreversible actions, no auto-rendering of attacker URLs.

## Common mistake

Believing a better system prompt, an injection *classifier*, or model alignment "solves" prompt injection. Detection is probabilistic and adversarially defeatable — given the trifecta, a sufficiently clever injection will eventually succeed. The only robust fixes are architectural: break a leg of the trifecta (isolate data, sandbox untrusted input, cut egress) so a successful injection still cannot exfiltrate.

## See also
- [[prompt-injection]] — the mechanism that supplies the "untrusted content" leg
- [[sub-agent-context-isolation]] — quarantining untrusted input in a separate context, the core CaMeL/dual-LLM defense
- [[model-context-protocol]] — the tool/connector layer where egress and data-access capabilities are granted
