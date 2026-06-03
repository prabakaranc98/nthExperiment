# Latent / Continuous Reasoning

**One-liner:** Reasoning in continuous latent space — feeding the model's last hidden state back as the next input embedding (Coconut) or adding recurrent depth at test time (looped transformers) — instead of decoding intermediate steps into discrete tokens, trading interpretability for higher information bandwidth per step.

## The key insight

Standard CoT bottlenecks reasoning through the vocabulary: each step is sampled to a discrete token, collapsing a high-dimensional hidden state h ∈ ℝ^d down to log₂|V| bits. Latent reasoning keeps the full continuous state.

**Coconut (Chain of Continuous Thought, Hao et al. 2024):** in "latent mode" the last hidden state is fed directly as the next-step input embedding, skipping the unembed→sample→embed round-trip:

    e_{t+1} = h_t      (latent mode; no token sampled)
    e_{t+1} = Embed(argmax W_U h_t)   (language mode)

Switch back to language mode to emit the final answer. Trained with a curriculum that progressively replaces written CoT steps with `<thought>` latent tokens.

**Looped / recurrent-depth (Geiping et al. 2025, "Huginn"):** apply a shared transformer block r times before unembedding, scaling effective depth at test time without more tokens:

    s_{k+1} = f_θ(s_k, e),  k = 1..r;   out = g(s_r)

r is sampled during training so depth can be dialed up at inference. This is recurrence-in-depth, distinct from token-level recurrence.

## Where it appears

- **Coconut (Meta, 2024)** — breadth-first-search-like latent reasoning; encodes multiple candidate next steps superposed in one continuous vector, outperforming CoT on planning tasks (ProsQA) with far fewer decoded tokens.
- **Huginn / latent recurrent-depth LM (2025)** — 3.5B model recurs a block up to ~32 times, matching much larger models on reasoning; test-time compute = number of loops, not token count.
- **Soft Thinking / mixture-of-embeddings (2025)** — feed back a probability-weighted average of token embeddings (a "soft token") rather than a hard sample, a training-free variant of the same idea.
- **Quiet-STaR / pause & filler tokens** — relatives that add latent compute via extra forward passes per token without explicit verbalization.

## Common mistake

Assuming latent reasoning is monitorable like CoT. It is the opposite: there are no human-readable intermediate tokens, so chain-of-thought faithfulness/monitoring breaks down entirely — you cannot read the trace. Also: looped recurrence (more depth per token) and Coconut (feeding hidden states back across steps) are different mechanisms; only the latter literally reasons "in" a continuous chain.

## See also
- [[chain-of-thought]] — the discrete-token baseline this replaces; latent reasoning is the "beyond tokens" frontier
- [[cot-faithfulness-monitorability]] — what you lose: latent traces are not human-readable
- [[test-time-compute-scaling]] — loop count / latent steps as the compute-scaling knob instead of generated length
