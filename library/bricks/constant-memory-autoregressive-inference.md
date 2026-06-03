# Constant-Memory Autoregressive Inference (No KV Cache)

**One-liner:** The headline serving advantage of recurrent/SSM models — a fixed-size hidden state replaces the linearly-growing transformer KV cache, giving O(1) memory and per-token decode cost regardless of context length.

## The key contrast

Transformer decode keeps every past key/value:
- KV cache size ∝ L (sequence length) → memory grows linearly, each new token attends over all L → per-token compute ∝ L, total decode ∝ L².

Recurrent/SSM decode carries a fixed state h_t ∈ ℝ^d (size independent of L):
- h_t = f(h_{t-1}, x_t),  y_t = g(h_t)
- For an SSM: h_t = Ā h_{t-1} + B̄ x_t,  y_t = C h_t. State is d_state × d_model, constant.
- Memory = O(1) in L, per-token cost = O(1), total decode = O(L).

The same model runs in two modes: a **parallel/chunked scan** during prefill (training-like, full sequence at once) and this **O(1) recurrence** during autoregressive decode.

## Where it appears

- **Mamba / Mamba-2 (S6, selective SSM)** — decode is a hardware-aware recurrence over the constant state; throughput is flat as context grows, unlike transformer KV.
- **RWKV, RetNet, GLA, DeltaNet** — linear-attention/fast-weight forms admit a recurrent decode with a fixed d×d state matrix (no per-token KV append).
- **Hybrid stacks (Jamba, Zamba, Samba, Falcon-Mamba)** — only the few interleaved attention layers grow a KV cache; SSM/linear layers stay O(1), drastically shrinking total cache.
- **Transformer-to-recurrent distillation (e.g. MOHAWK, Mamba-in-Llama)** — converts a pretrained transformer to a recurrent form precisely to get constant-memory serving.

## Common mistake

Equating "constant memory" with "free unlimited context." The fixed state is also fixed *capacity*: you trade KV growth for a finite memory bottleneck, so exact long-range recall degrades (the associative-recall problem). Hybrids add a little attention back specifically to recover what the compressed state forgets.

## See also
- [[kv-cache]] — the linearly-growing transformer cost this avoids
- [[selective-state-space-models-mamba]] — the canonical constant-state decode in practice
- [[associative-recall-the-recall-state-size]] — why a fixed state limits exact recall
