# Associative Recall & the Recall-State-Size Tradeoff

**One-liner:** MQAR (multi-query associative recall) is the diagnostic for whether a sub-quadratic model can store arbitrary in-context key→value bindings; a fixed-size recurrent state imposes a hard recall ceiling that scales with state size, while attention's growing KV-cache buys unbounded recall at O(N) memory.

## The task / the bound

**MQAR (Arora et al., "Zoology"/"Based", 2023-24):** the prompt interleaves key-value pairs then queries already-seen keys; the model must emit the bound value:
```
A 4  B 3  C 6 ... | A → 4 ?   C → 6 ?
```
Single-query AR (induction-head style "...A 4 ... A → ?") is easy; the *multi-query* version stresses storage of many simultaneous bindings.

**The tradeoff.** A model with recurrent state of d_state bits can recall at most ~d_state bits of binding information — recall accuracy degrades once #pairs × bits/pair exceeds state capacity. Empirically (Zoology): recall ∝ d_state, and there is a Pareto frontier of **recall vs. recurrent-state size**. Attention sits at the unbounded-recall extreme (state = full KV-cache, O(N·d)); gated linear-attention / SSMs trade a fixed O(d_state) state for a capacity wall.

## Where it appears

- **Based / Zoology (Arora et al., 2024)** — used MQAR to show linear attention fails recall at small state; motivates mixing a small sliding-window softmax head (exact local recall) with linear attention (cheap global).
- **Mamba / Mamba-2 (Gu & Dao, 2023-24)** — selective SSM; input-dependent gating lets the fixed-state recurrence *choose* what to store, lifting the recall frontier vs. data-independent (LTI) SSMs.
- **Jamba, Zamba, MiniMax-01, hybrid stacks (2024-26)** — interleave a few full-attention layers with many SSM/GLA layers precisely so the attention layers handle exact recall while SSM layers carry the bulk cheaply.
- **DeltaNet / Gated DeltaNet** — delta-rule fast-weight updates raise effective recall capacity per state bit by overwriting stale key-value associations.

## Common mistake

Reading "linear attention matches transformers on perplexity" as "it can recall." Perplexity is dominated by easy local tokens; MQAR isolates the recall subskill that PPL hides. The flip side: assuming a bigger fixed state is "free" — it raises the recall ceiling but never makes it unbounded, so any constant-state model has an in-context length past which arbitrary lookup must fail.

## See also
- [[induction-heads]] — the attention mechanism that solves single-query recall exactly
- [[gated-linear-attention-data-dependent-decay]] — data-dependent gating is how fixed-state models climb the recall frontier
- [[kv-cache]] — attention's unbounded-recall extreme; the O(N) memory cost the tradeoff weighs against
