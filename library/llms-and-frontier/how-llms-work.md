# How LLMs Work

*The complete picture: from raw text to a model that can reason.*

---

## The one-sentence version

An LLM is a function that takes a sequence of tokens and outputs a probability distribution over the next token. It was trained to predict the next token on a massive corpus of text. Almost everything else follows from that.

---

## Step 1: Tokenization

Text is converted to integers (tokens) before the model sees it. A tokenizer splits text into subword units using **BPE (Byte-Pair Encoding)**:

1. Start with individual characters as the vocabulary
2. Repeatedly merge the most frequent pair of adjacent tokens into a new token
3. Stop when you reach the target vocabulary size (~32K–128K tokens)

Why subwords? Balances vocabulary size against sequence length. Whole-word tokenization = huge vocabulary. Character tokenization = very long sequences. Subwords are in between.

**Important:** "token" ≠ "word". "tokenization" might be ["token", "ization"]. Numbers are often split digit-by-digit. Spaces matter.

---

## Step 2: Pretraining — predicting the next token

The training objective is simple: given tokens [t₁, t₂, ..., tₙ], predict tₙ₊₁.

**The loss:** cross-entropy between the model's predicted distribution and the actual next token:
```
L = -log P(tₙ₊₁ | t₁, ..., tₙ)
```

Do this across billions of (context, next-token) pairs. The model learns, by necessity, to understand language, facts, reasoning patterns, and code — because all of these help it predict the next token more accurately.

**Training data:** 10–15 trillion tokens for frontier models (Llama 3 used 15T). Mostly web text (Common Crawl), filtered and deduplicated, mixed with books, code, Wikipedia, math, and scientific papers.

---

## Step 3: What the model learns to represent

A model trained only on next-token prediction learns surprisingly rich structure:
- **Syntax and grammar** — because they're predictable
- **Factual knowledge** — because it's repeated across the web
- **Reasoning patterns** — because logical sequences are more predictable than random ones
- **In-context learning** — patterns in the prompt that shift what's expected

The key insight: **compression is understanding**. If you can predict text well, you must have understood it.

---

## Step 4: Post-training — shaping behavior

A pretrained model is a powerful text predictor, not a helpful assistant. Post-training shapes it:

1. **Supervised Fine-tuning (SFT):** train on curated (instruction, response) pairs. The model learns to follow instructions.

2. **Preference optimization (RLHF / DPO):** compare model outputs, train to prefer better ones. The model learns to be helpful, harmless, and honest.

3. **RLVR (reasoning):** for math/code, use verifiable rewards (right/wrong answer). The model learns to reason.

---

## The full pipeline

```
Raw text
  → Tokenize (BPE)
  → Pretrain on next-token prediction (10T+ tokens)
  → SFT on instruction data
  → DPO / RLHF / RLVR on preferences / verifiable rewards
  → Deploy (with quantization, KV cache, inference optimizations)
```

---

## Key numbers (frontier models, 2025)

| Property | Range |
|----------|-------|
| Parameters | 7B–1T+ (MoE: sparse activation) |
| Pretraining tokens | 10T–15T |
| Context window | 128K–10M tokens |
| Vocabulary size | 32K–128K tokens |
| Training compute | 10²³–10²⁵ FLOPs |
| Training cost | $1M–$100M+ |

---

## Three things people misunderstand

1. **"It just memorizes things."** No — it also generalizes. It can answer questions about combinations of facts it's never seen exactly.

2. **"Bigger is always better."** At a fixed compute budget, you often get a better model by training a smaller model on more tokens (Chinchilla insight). The *inference-optimal* model is often undertrained relative to compute-optimal.

3. **"It doesn't understand, it just predicts tokens."** This debate is unsettled. What's clear: it builds rich internal representations that support diverse tasks. Whether that's "understanding" is a philosophical question.
