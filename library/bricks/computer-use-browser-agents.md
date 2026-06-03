# Computer Use / Browser Agents

**One-liner:** Agents that perceive a GUI as screenshots and/or accessibility/DOM trees and act through low-level mouse/keyboard primitives (click(x,y), type, scroll, key), looping perceive -> reason -> act until a task is done — operating arbitrary software the way a human would instead of via bespoke APIs.

## The loop / action interface

Closed-loop ReAct-style control over a coordinate action space:

```
s_0 = screenshot()                      # + optional a11y/DOM tree, prior actions
while not done:
    a_t = policy(instruction, s_t, history)   # VLM emits a tool call
    # a_t in {click(x,y), double_click, drag, type(text),
    #         key("ctrl+c"), scroll(dx,dy), wait, screenshot, done}
    execute(a_t)                          # via OS automation / CDP / Playwright
    s_{t+1} = screenshot()                # observe new state, append to history
```

Two grounding regimes:
- **Pixel grounding** (Claude Computer Use, UI-TARS, OpenAI CUA/Operator): VLM outputs raw (x,y) coordinates from the screenshot. Universal, app-agnostic, but coordinate precision is the bottleneck.
- **Structured grounding** (DOM/a11y trees, set-of-marks): elements are labeled with IDs/numbered overlays; the model picks an element_id instead of a pixel. More reliable in browsers, breaks on canvas/non-DOM UIs.

History is usually image+action interleaving with old screenshots downsampled or dropped to fit context.

## Where it appears

- **Claude Computer Use (Anthropic, Oct 2024)** — beta tool API exposing screenshot + cursor/keyboard actions; pixel-coordinate grounding in a sandboxed VM.
- **OpenAI Operator / CUA (computer-using agent, Jan 2025)** — browser-focused agent built on GPT-4o-class vision + RL; benchmarked on WebArena/WebVoyager.
- **UI-TARS / Aguvis / SeeClick** — end-to-end native GUI VLMs trained on grounding + trajectory data; UI-TARS unifies perception, grounding, and action in one model.
- **Benchmarks** — OSWorld, WebArena/VisualWebArena, WebVoyager, Mind2Web, AndroidWorld measure task success on real OS/web environments.

## Common mistake

Treating it as pure perception/grounding accuracy. The hard, security-critical failure is **the lethal trifecta**: an agent with access to private data, exposure to untrusted content (web pages, emails, screen text), and the ability to exfiltrate. A page or popup can carry an injected instruction the model executes — coordinate accuracy near 100% does not make a computer-use agent safe to point at your real machine.

## See also
- [[react]] — the perceive-reason-act loop computer-use agents run
- [[agent-security-the-lethal-trifecta]] — the core deployment risk for screen/web agents
- [[function-tool-calling]] — the mechanism each click/type action is emitted through
