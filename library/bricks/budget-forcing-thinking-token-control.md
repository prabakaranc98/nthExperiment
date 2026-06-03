# Budget Forcing / Thinking-Token Control

**One-liner:** A test-time decoding intervention that explicitly sets the reasoning-token budget — either capping the thinking trace by force-injecting an end-of-thinking delimiter, or extending it by suppressing that delimiter and appending "Wait" — to slide a single model along the accuracy-vs-latency curve (s1-style).

## The mechanism

Reasoning is bracketed by delimiters, e.g. `<think> ... </think>` then the answer. Budget forcing manipulates *when* `</think>` is emitted:

- **Cap (force-stop):** once the trace hits N thinking tokens, suppress further generation and inject `</think>` + an answer-eliciting string (e.g. "Final Answer:"), forcing the model to commit early. Lower N → lower latency, lower accuracy.
- **Extend (force-continue):** when the model *wants* to emit `</think>`, ban that token and append a continuation string ("Wait", "Hmm", "Let me reconsider"). The model second-guesses itself and keeps reasoning. More appends → more compute → typically higher accuracy, until it plateaus/degrades.

Pseudocode (extend):
```
while step < max_steps:
    tok = sample()
    if tok == END_THINK and forced_extensions < k:
        ban(END_THINK); emit("\nWait")   # keep thinking
        forced_extensions += 1
    else:
        emit(tok)
```

Sweeping N traces a monotone-ish test-time scaling curve: accuracy increases with thinking-token budget. s1 reports this curve is roughly *log-linear* in the budget before saturating.

## Where it appears

- **s1 / s1.1 (Muennighoff et al., 2025)** — coined "budget forcing"; 1k-example SFT on reasoning traces (s1K) + the Wait-append trick, getting o1-preview-level math from a small model with minimal RL.
- **Reasoning models with explicit budgets** — Claude (extended-thinking `budget_tokens`), Gemini 2.5 (`thinkingBudget`), and OpenAI o-series `reasoning_effort` expose the user-facing knob; internally a cap/early-commit on the reasoning trace.
- **Overthinking mitigation** — capping is used to kill runaway traces on easy prompts; pairs with per-query *adaptive* budget allocation.

## Common mistake

Assuming "Wait"-extension monotonically improves accuracy, so more is always better. It saturates and then *hurts* — over-long traces drift, self-doubt into wrong answers, or loop. The right budget is query-difficulty dependent; a fixed large N wastes compute on easy items and can degrade them.

## See also
- [[test-time-compute-scaling]] — budget forcing is the simplest knob for spending test-time compute
- [[compute-optimal-test-time-allocation]] — choosing N per-query instead of a fixed cap
- [[long-reasoning-chains]] — the reasoning traces whose length this controls
