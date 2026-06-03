# The Transformer

*Built from scratch. Every component explained before it's used.*

---

## The problem transformers solve

Before transformers, sequence models (RNNs, LSTMs) processed tokens one at a time. Each token's representation depended on all previous tokens through a hidden state. Two problems:
1. **Hard to parallelize** — you must process token 1 before token 2
2. **Hard to model long-range dependencies** — information from early tokens fades as the sequence grows

Transformers solve both by processing all tokens simultaneously, using **attention** to let every token directly look at every other token.

---

## Attention — the core idea

Given a sequence of tokens, attention lets each token ask: *"which other tokens are most relevant to me?"*

Three vectors per token:
- **Query (Q)**: what this token is looking for
- **Key (K)**: what this token offers to others looking at it
- **Value (V)**: the actual information to extract if this token is selected

**Scaled dot-product attention:**

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

Step by step:
1. Compute Q·Kᵀ — a score for how relevant each key is to each query
2. Divide by √dₖ (dimension of keys) — prevents scores from being too large (softmax saturates)
3. Apply softmax — turns scores into probabilities (attention weights sum to 1)
4. Multiply by V — take a weighted average of the values

The output for each token is a weighted mix of all other tokens' values, where the weights reflect relevance.

---

## Multi-head attention

Instead of one set of Q, K, V projections, use h heads in parallel:
- Each head projects to a lower dimension (dₖ = d_model / h)
- Each head can attend to different types of relationships
- Outputs are concatenated and projected back

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ
where headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᵏ, V·Wᵢᵛ)
```

---

## The transformer block

A transformer block contains:
1. **Multi-head self-attention** — tokens attend to each other
2. **Residual connection** — add the input back: x + Attention(x)
3. **Layer normalization** — normalize the residual
4. **Feed-forward network** — two linear layers with an activation (SwiGLU in modern LLMs)
5. **Another residual connection + layer norm**

The residual connections are critical: they let gradients flow directly to earlier layers, making deep networks trainable.

---

## Positional encoding

Attention is *permutation invariant* — it doesn't know the order of tokens. Positional encodings inject position information.

**Original transformer:** sinusoidal encodings added to embeddings.

**Modern LLMs use RoPE (Rotary Position Embedding):** instead of adding positions to embeddings, rotate the query and key vectors by an angle proportional to their position before computing attention. This means the dot product between Q and K naturally depends on their *relative* position, not absolute positions. This generalizes better to longer sequences than the model was trained on.

---

## The full decoder-only transformer (GPT-style)

Modern LLMs are decoder-only: they generate tokens left-to-right.

Key difference: **causal masking** — when computing attention, each token can only attend to *previous* tokens (including itself). This prevents the model from "cheating" by looking at future tokens during training.

```
Input tokens
    → Embedding (token index → vector)
    → Add positional encoding (RoPE)
    → N × Transformer blocks
        → Causal self-attention
        → Residual + LayerNorm
        → FFN (SwiGLU)
        → Residual + LayerNorm
    → Final LayerNorm
    → Linear projection to vocabulary size
    → Softmax → probability over next token
```

---

## Grouped Query Attention (GQA)

**The memory problem:** the KV cache (stored keys and values for already-processed tokens) grows linearly with sequence length. For long contexts, it's the bottleneck.

**MHA (Multi-Head Attention):** each head has its own K, V. h heads × sequence length × d_head × 2 = lots of memory.

**GQA:** multiple query heads share the same K, V head. Reduces KV cache by h/g where g is the number of groups. Almost no quality loss in practice.

**MQA (Multi-Query Attention):** extreme case — all query heads share one K, V. Fastest, slightly more quality loss.

---

## What the transformer can and cannot do

**Can:**
- Model arbitrary long-range dependencies (if context fits in the window)
- Be parallelized fully during training
- Scale reliably with more data and parameters

**Cannot (by construction):**
- Process sequences longer than its context window without tricks
- Run in constant memory (KV cache grows with sequence length)
- Easily model strictly sequential/recurrent dependencies

---

## Key numbers to know

| Component | Modern setting |
|-----------|---------------|
| Attention | Multi-head, GQA or MLA |
| Normalization | RMSNorm, pre-norm (before attention + FFN) |
| FFN activation | SwiGLU |
| Positional encoding | RoPE |
| Context length | 128K–1M+ tokens (frontier models) |
| d_model | 4096–16384 (7B–70B models) |
| Layers | 32–96 |
