# Self-Reflection / Reflexion

**One-liner:** The agent verbally critiques its own failed output/trajectory and stores that critique in context for the retry — turning a scalar/environment signal into natural-language feedback that steers the next attempt, with zero weight updates.

## The loop

Reflexion (Shinn et al., NeurIPS 2023) factors the agent into three roles:

```
for trial t = 1..K:
    τ_t      = Actor(task, memory)         # produce trajectory / answer
    r_t      = Evaluator(τ_t)              # scalar/binary: unit test, exact-match, env reward
    if r_t == success: break
    refl_t   = SelfReflection(τ_t, r_t)    # LLM writes verbal post-mortem: WHY it failed
    memory.append(refl_t)                  # episodic buffer, fed into next prompt
```

The reflection text (not the score) is the gradient surrogate: it diagnoses the cause ("I forgot to handle the empty-list case") and is prepended to the next attempt. Memory is bounded (last 1-3 reflections) to fit context.

## Where it appears

- **Reflexion (2023)** — HumanEval/coding, ALFWorld, HotpotQA; ~11-20pt gains by retrying with self-written feedback against a verifier
- **Self-Refine (Madaan et al., 2023)** — same idea without an external evaluator: model generates → critiques itself → revises, iterated
- **Agentic coding loops (2024-2026)** — SWE-agent / Claude-Code-style harnesses reflect on test/CI failures before re-editing
- **Reasoning-RL data gen** — reflection traces are distilled into models (STaR-style) so the behavior becomes weight-baked, not just in-context

## Common mistake

Assuming reflection helps *without a reliable verifier*. The lift comes almost entirely from the evaluator signal; with self-judged correctness on tasks the model can't verify, reflection often degenerates into confident wrong self-affirmation and stalls or worsens (intrinsic self-correction is weak). It is a test-time-compute trade, not free accuracy.

## See also
- [[self-correction-reflection]] — single-turn revise-your-own-answer; Reflexion is its multi-trial, memory-augmented form
- [[react]] — the reason+act trajectory format Reflexion reflects over
- [[self-improvement-star-bootstrapping]] — distilling reflection/retry traces into weights
