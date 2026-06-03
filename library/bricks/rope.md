# RoPE (Rotary Position Embedding)

**One-liner:** Inject position by rotating each Q/K vector's 2D feature pairs by an angle proportional to absolute position m, so the dot product qₘ·kₙ depends only on the relative offset (m−n) — no learned positional table, no additive bias.

## The formula / definition

Split a head's d-dim Q/K vector into d/2 pairs. Pair i gets frequency θᵢ = base^(−2i/d) (base = 10000 originally). At position m, rotate pair i by angle mθᵢ:

```
[x₂ᵢ ]   [cos(mθᵢ)  −sin(mθᵢ)] [x₂ᵢ ]
[x₂ᵢ₊₁] = [sin(mθᵢ)   cos(mθᵢ)] [x₂ᵢ₊₁]
```

Apply this rotation R(m) to q (at pos m) and R(n) to k (at pos n). Then:

```
⟨R(m)q, R(n)k⟩ = qᵀ R(m)ᵀR(n) k = qᵀ R(n−m) k
```

Rotations compose as R(m)ᵀR(n) = R(n−m), so the attention logit is a pure function of relative position. Applied to Q and K only, after the projection, before the dot product — never to V.

## Where it appears

- **RoFormer (Su et al., 2021)** — original proposal; the now-default position scheme.
- **LLaMA / Llama 2/3, Mistral, Qwen, Gemma, DeepSeek** — RoPE on every layer is the standard transformer recipe in 2024-2026.
- **Context extension** — NTK-aware / "YaRN" / linear position interpolation rescale θ (raise the base, e.g. 10000 → 500000 in Llama 3, or interpolate positions) to extend context far beyond the trained length with little fine-tuning.
- **GQA stacks / FlashAttention kernels** — RoPE is fused into the QK path; it costs almost nothing.

## Common mistake

Thinking RoPE adds a position vector. It does not — it is a *multiplicative* rotation applied to Q and K (not V), and it carries no learned parameters. A second mistake: extending context by just feeding longer sequences. Without rescaling θ (interpolation/YaRN), the high-frequency pairs see angles far outside the training distribution and attention degrades sharply past the trained length.

## See also
- [[gqa]] — RoPE is applied per-head inside the same QK path GQA shares across heads
- [[kv-cache]] — cached K already has rotation baked in, so positions stay correct at decode
- [[flash-attention]] — RoPE is fused into the QK computation in production attention kernels
