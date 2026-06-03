# Transformer Architecture — Revision Sheet

## The building block

```
Input tokens → Embedding → (+ RoPE positional encoding)
    ↓
[N × Transformer Block]
    1. RMSNorm
    2. Causal Self-Attention (GQA or MLA)
       Q = x·Wq, K = x·Wk, V = x·Wv
       Attn = softmax(QKᵀ/√dₖ) · V  [causal mask: no future tokens]
    3. Residual: x = x + Attn(x)
    4. RMSNorm
    5. FFN: SwiGLU(x·W₁, x·W₂) · W₃  [gated MLP]
    6. Residual: x = x + FFN(x)
    ↓
Final RMSNorm → Linear (→ vocab size) → Softmax → P(next token)
```

## Attention variants (KV cache trade-off)

| Method | KV heads | Memory | Quality |
|--------|----------|--------|---------|
| MHA | h (same as Q) | Full | Best |
| GQA | g < h (shared groups) | Reduced | Near-MHA |
| MQA | 1 (fully shared) | Minimal | Slight loss |
| MLA | Low-rank compressed | Minimal | Near-MHA |

## Key design choices (modern LLMs)

| Component | Classic | Modern (2024+) |
|-----------|---------|---------------|
| Normalization | Post-norm BatchNorm | Pre-norm RMSNorm |
| Positional | Sinusoidal (absolute) | RoPE (relative) |
| FFN activation | ReLU | SwiGLU |
| Attention | MHA | GQA or MLA |
| Context | 2K tokens | 128K–10M tokens |

## Scaling (rough rules of thumb)

- d_model ≈ 128 × (num_layers)^0.5
- d_ffn ≈ 4 × d_model (SwiGLU: 2/3 × 4 × d_model per gate)
- Parameters: ~12 × d_model² × num_layers (dense)
- FLOPs per forward pass: ~6N (N = parameters)

## Residual stream view

Think of the residual stream as a "communication bus". Each attention head and FFN layer reads from and writes to it. The residual connection ensures information can bypass any layer unchanged.
