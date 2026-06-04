# How LLMs Work

*The complete picture: from raw text to a model that can reason.*

---

## The one-sentence version

An LLM is a function that maps a sequence of tokens to a probability distribution over the next token. It was trained to predict that next token on a massive corpus of text. Almost everything else follows from that.

---

## Step 1: Tokenization

Text becomes integers (tokens) before the model sees it. A tokenizer splits text into subword units, most commonly with **BPE (Byte-Pair Encoding)**:

1. Start with bytes/characters as the base vocabulary.
2. Repeatedly merge the most frequent adjacent pair into a new token.
3. Stop at the target vocabulary size (~32K–256K tokens).

**Why subwords?** They trade off vocabulary size against sequence length. Whole-word vocabularies are huge; character vocabularies make sequences very long. Subwords sit in between.

**Watch out:**
- "token" ≠ "word" — `"tokenization"` might split as `["token", "ization"]`.
- Numbers are often split digit-by-digit (or in fixed chunks), which is why naive arithmetic is hard.
- Leading spaces are part of the token; `"the"` and `" the"` are different.

---

## Step 2: Pretraining — predicting the next token

The objective is simple: given tokens [t₁, …, tₙ], predict tₙ₊₁.

**Loss** — cross-entropy between the predicted distribution and the actual next token:

```
L = -log P(tₙ₊₁ | t₁, ..., tₙ)
```

Repeat across trillions of (context, next-token) pairs. To minimize this loss the model is forced to internalize language, facts, reasoning patterns, and code — all of which make the next token more predictable.

**Training data:** 15T+ tokens for frontier models (Llama 3 used 15T; newer flagships train on more). Mostly filtered, deduplicated web text, mixed with books, code, math, and scientific papers — increasingly supplemented with synthetic and model-generated data.

---

## Step 3: What the model learns to represent

Next-token prediction alone produces surprisingly rich structure:

- **Syntax and grammar** — highly predictable, so learned early.
- **Factual knowledge** — repeated across the corpus.
- **Reasoning patterns** — logical sequences are more predictable than random ones.
- **In-context learning** — cues in the prompt that shift what comes next.

The core idea: **compression is understanding**. Predicting text well requires having modeled what generated it.

---

## Step 4: Post-training — shaping behavior

A pretrained model is a powerful text predictor, not a helpful assistant. Post-training reshapes it:

| Stage | What it does | Signal |
|-------|--------------|--------|
| **SFT** (Supervised Fine-tuning) | Learns to follow instructions | Curated (instruction, response) pairs |
| **Preference optimization** (RLHF / DPO) | Learns to be helpful, harmless, honest | Human/AI comparisons between outputs |
| **RLVR** (Reinforcement Learning from Verifiable Rewards) | Learns to reason | Automatic right/wrong checks on math & code |

RLVR is what drives the "reasoning model" line (long chains of thought trained against verifiable rewards) that defines much of 2025–2026 frontier practice.

---

## The full pipeline

```
Raw text
  → Tokenize (BPE)
  → Pretrain on next-token prediction (15T+ tokens)
  → SFT on instruction data
  → RLHF / DPO on preferences  +  RLVR on verifiable rewards
  → Deploy (quantization, KV cache, speculative decoding)
```

---

## Key numbers (frontier models, 2025–2026)

| Property | Range |
|----------|-------|
| Parameters | 7B–1T+ (MoE: only a fraction active per token) |
| Pretraining tokens | 15T+ |
| Context window | 128K–1M+ tokens |
| Vocabulary size | 32K–256K tokens |
| Training compute | 10²³–10²⁶ FLOPs |
| Training cost | $1M–$100M+ |

---

## Three things people misunderstand

1. **"It just memorizes."** It also generalizes — answering questions about combinations of facts it never saw together verbatim.

2. **"Bigger is always better."** At a fixed compute budget, a smaller model trained on more tokens often wins (the Chinchilla result). And because most cost is at inference, the *inference-optimal* model is usually trained well past compute-optimal — smaller and longer-trained on purpose.

3. **"It doesn't understand, it just predicts tokens."** Unsettled. What is clear: it builds internal representations rich enough to support diverse tasks. Whether that counts as "understanding" is a philosophical question, not a technical one.

---

*Related: see the [concept library index](../bricks/README.md) for tokenization, scaling laws, and inference-optimization bricks.*
