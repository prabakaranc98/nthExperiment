# Attention Sinks

**One-liner:** Softmax forces attention weights to sum to 1, so when no token is relevant, heads dump excess mass onto the first token(s) — these "sinks" are load-bearing for stability, and dropping them (e.g. in naive KV-cache eviction or windowing) collapses generation.

## The key insight

Softmax attention has no "null" option: scores normalize to a probability distribution that must sum to 1, even when the query attends to nothing useful. The model learns to park this excess probability on always-visible tokens — overwhelmingly position 0 (the BOS/first token). The sink token typically has near-zero value-vector content, so attending to it ≈ a no-op, but it absorbs mass that would otherwise distort the weighted sum.

StreamingLLM (Xiao et al., ICLR 2024) diagnosed this: a sliding-window KV cache that evicts the first tokens causes a perplexity blowup. Fix = keep a few (≈4) initial-token KV entries as permanent sinks + the recent window:

```
KV cache = [sink tokens (first ~4, never evicted)] ++ [rolling window of last L tokens]
```

Architectural fix: add a learned, content-free sink. Either a dedicated register/sink token prepended at train time, or an off-by-one / "softmax_1" reformulation that gives softmax an explicit escape valve:

```
softmax_1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))   # +1 lets all weights be ~0
```

GPT-OSS (2025) ships a per-head learnable additive sink logit in the attention denominator.

## Where it appears

- **StreamingLLM** — keep first-token KV + sliding window for infinite-length streaming with bounded cache
- **KV-cache eviction (H2O, Scissorhands, FastGen)** — must protect sink/heavy-hitter tokens or quality craters
- **Attention sink registers** — ViTs (Darcet et al., 2024 "Vision Transformers Need Registers") add register tokens to soak high-norm sink artifacts; LLMs adopt the same
- **GPT-OSS / Gemma / modern LLMs** — explicit learnable sink logits baked into the attention kernel
- **Quantization** — sink tokens carry massive activation norms; outlier-aware schemes treat them specially

## Common mistake

Thinking the sink is "wasted" computation you can prune. It is a functional pressure-release valve for the softmax constraint — evict it (or naively re-roll positions so the model loses its BOS anchor) and attention mass redistributes onto recent tokens, corrupting the hidden state and spiking perplexity.

## See also
- [[kv-cache-compression-eviction]] — eviction policies must preserve sink tokens to stay coherent
- [[softmax]] — the sum-to-1 constraint is the root cause; softmax_1 removes it
- [[sliding-window-local-attention]] — windowing breaks without retained sinks (StreamingLLM's setting)
