# The Transformer

*Built from scratch. Every component explained before it's used.*

---

## The problem transformers solve

Before transformers, sequence models (RNNs, LSTMs) processed tokens one at a time, each token's representation depending on all previous tokens through a single hidden state. Two problems:

1. **Hard to parallelize** — token 2 cannot be processed before token 1.
2. **Hard to model long-range dependencies** — information from early tokens fades as the sequence grows.

Transformers fix both by processing all tokens simultaneously and using **attention** to let every token look directly at every other token.

---

## Attention — the core idea

Attention lets each token ask: *"which other tokens are most relevant to me?"* Each token produces three vectors:

| Vector | Role |
|--------|------|
| **Query (Q)** | what this token is looking for |
| **Key (K)** | what this token offers to others looking at it |
| **Value (V)** | the information to extract if this token is selected |

**Scaled dot-product attention:**

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

Step by step:

1. **QKᵀ** — score how relevant each key is to each query.
2. **÷ √dₖ** — scale by the key dimension so large dot products don't saturate the softmax.
3. **softmax** — turn scores into weights that sum to 1.
4. **· V** — take the weighted average of the values.

Each token's output is a weighted mix of all tokens' values, where the weights reflect relevance.

---

## Multi-head attention

Instead of one set of Q, K, V projections, use `h` heads in parallel:

- Each head projects to a lower dimension (`dₖ = d_model / h`).
- Each head can specialize in a different type of relationship.
- Outputs are concatenated and projected back to `d_model`.

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ
where  headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᵏ, V·Wᵢᵛ)
```

---

## The transformer block

A block stacks two sublayers, each wrapped in a residual connection and normalization:

1. **Multi-head self-attention** — tokens attend to each other.
2. **Residual + norm** — `x + Attention(x)`, then normalize.
3. **Feed-forward network (FFN)** — two linear layers with an activation (SwiGLU in modern LLMs).
4. **Residual + norm** again.

Modern LLMs use **pre-norm**: normalization is applied *before* each sublayer, which stabilizes training of deep stacks. Residual connections are critical — they give gradients a direct path to earlier layers, making deep networks trainable.

---

## Positional encoding

Attention is *permutation invariant* — by itself it has no notion of token order. Positional encodings inject that information.

- **Original transformer:** fixed sinusoidal encodings added to the embeddings.
- **Modern LLMs (RoPE, Rotary Position Embedding):** rotate the query and key vectors by an angle proportional to their position *before* computing attention. The Q·K dot product then depends on the *relative* distance between tokens rather than absolute positions, which extrapolates better to sequences longer than seen in training.

---

## The full decoder-only transformer (GPT-style)

Modern LLMs are decoder-only: they generate tokens left to right.

The key mechanism is **causal masking** — each token may attend only to *previous* tokens (and itself). This stops the model from "cheating" by reading future tokens during training.

```
Input tokens
    → Embedding (token index → vector)
    → Add positional encoding (RoPE)
    → N × Transformer blocks
        → Causal self-attention
        → Residual + norm
        → FFN (SwiGLU)
        → Residual + norm
    → Final norm
    → Linear projection to vocabulary size
    → Softmax → probability over next token
```

---

## Attention variants and the KV-cache bottleneck

During generation, the **KV cache** (stored keys and values for tokens already processed) grows linearly with sequence length. For long contexts it dominates memory, so attention variants trade head independence for a smaller cache.

| Variant | K/V sharing | KV cache | Quality |
|---------|-------------|----------|---------|
| **MHA** (Multi-Head) | each query head has its own K, V | largest | baseline |
| **GQA** (Grouped-Query) | query heads share K, V within `g` groups | ÷ (h/g) | ~no loss |
| **MQA** (Multi-Query) | all query heads share one K, V | smallest | slight loss |
| **MLA** (Multi-head Latent) | K, V compressed into a low-rank latent | very small | competitive |

GQA is the common default for frontier open-weight models; MLA (popularized by DeepSeek) compresses the cache further while staying competitive on quality.

---

## What the transformer can and cannot do

**Can:**

- Model arbitrary long-range dependencies, as long as they fit in the context window.
- Be parallelized fully during training.
- Scale predictably with more data and parameters.

**Cannot (by construction):**

- Process sequences longer than its context window without extra tricks.
- Run in constant memory — the KV cache grows with sequence length.
- Naturally express strictly sequential / recurrent dependencies.

---

## Key numbers to know (2025–2026)

| Component | Modern setting |
|-----------|----------------|
| Attention | Multi-head with GQA or MLA |
| Normalization | RMSNorm, pre-norm (before attention and FFN) |
| FFN | SwiGLU; often Mixture-of-Experts (sparse FFN) at scale |
| Positional encoding | RoPE (frequently with context-extension scaling) |
| Context length | 128K–1M+ tokens (frontier models) |
| d_model | 4096–16384 (≈7B–70B dense models) |
| Layers | 32–96 |

See the [concept-library index](../bricks/README.md) for related building blocks.
