# Constrained / Structured Decoding

**One-liner:** At each decode step, mask the next-token logits against a grammar (JSON schema, regex, CFG) so only tokens that keep the output on a valid prefix can be sampled — output is syntactically valid by construction, a hard requirement for reliable tool-calling.

## The mechanism

Compile the grammar into a finite-state machine / pushdown automaton over the tokenizer's vocabulary. At each step, given the current FSM state q, precompute the allowed token set A(q) ⊆ V, then:

  logits[i] ← −∞   for all i ∉ A(q)
  sample from softmax(logits); advance q with the chosen token.

Outlines (Willard & Louf, 2023) builds an FSM-over-vocabulary index so the allowed-token mask is an O(1) dict lookup per state, not a per-token regex scan — the mask compile is amortized once per schema. XGrammar (2024) handles full CFGs with a byte-level pushdown automaton, splitting tokens into context-independent (precomputable mask) vs context-dependent (runtime stack check), overlapping mask computation with GPU decode for near-zero overhead.

## Where it appears

- **Outlines / XGrammar / llguidance** — regex + JSON-schema + EBNF-grammar masking; XGrammar is the default backend in vLLM, SGLang, TensorRT-LLM.
- **OpenAI / Anthropic JSON mode & Structured Outputs** — provider-side constrained decoding guaranteeing schema-conformant tool-call arguments and response objects.
- **Function/tool calling** — the call args (a typed JSON object) are emitted under schema constraint so the parse never fails; the backbone of agentic loops.

## Common mistake

Believing valid *syntax* implies valid *semantics* or unbiased *content*. Masking only guarantees the string parses; it cannot make a field truthful, and it actively distorts the distribution — forcing a token the model assigned low probability (e.g., closing a JSON early) can degrade reasoning. Best practice: let the model reason in free text (CoT), then constrain only the final structured span.

## See also
- [[function-tool-calling]] — schema-constrained args are what make tool calls reliably parseable
- [[decoding-sampling-strategies]] — masking composes with temperature/top-p; it edits logits before sampling
- [[gumbel-softmax-straight-through-estimator]] — both manipulate the categorical token distribution, one for validity, one for differentiability
