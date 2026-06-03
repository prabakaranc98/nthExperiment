# Tokenizer Design (BPE / Byte-Level / Tokenizer-Free)

**One-liner:** The choice of subword algorithm (BPE, byte-level BPE, Unigram-LM) and vocab size V fixes compression (bytes/token), fertility (tokens/word), and the model's representable atoms — with byte-latent and 200K+ vocabs reopening choices long treated as settled.

## The key quantities

- **Compression ratio** = bytes / tokens. Higher = each token carries more text -> shorter sequences, cheaper O(N²) attention, more text per context window.
- **Fertility** = tokens / word (in a reference unit, e.g. whitespace word or NFC grapheme). >1; high fertility on a language/script means that language is "taxed" — more tokens, worse loss, higher cost. Measured per-language to expose multilingual unfairness.
- **Bits-per-byte** normalizes loss across tokenizers: `bpb = (loss_nats / ln2) * (tokens / bytes)`. Always compare models in bpb, never raw per-token perplexity (perplexity is vocab/compression-dependent and not comparable).
- **V vs sequence length tradeoff:** larger V -> better compression -> shorter D in tokens, but a bigger embed/unembed matrix (V·d params, V·d·N softmax FLOPs) and more rare tokens with under-trained embeddings.

## The three algorithm families

- **BPE / byte-level BPE (BBPE):** greedy rank-ordered merges (see [[bpe]]). BBPE merges over raw UTF-8 bytes -> zero OOV, no `<unk>`. GPT/Llama/Mistral default.
- **Unigram-LM (SentencePiece):** start from a large candidate set, prune via EM to maximize corpus likelihood under a unigram model; encode by Viterbi (probabilistic, supports sampling for subword regularization). T5, ALBERT, mBART, Gemma.
- **Tokenizer-free / byte-latent:** no fixed vocab. BLT (Byte Latent Transformer, 2024) groups bytes into dynamic patches by next-byte entropy and runs the big transformer over patches; MegaByte/MambaByte/Charformer are earlier byte models. Trades the frozen merge table for learned, content-adaptive segmentation.

## Where it appears

- GPT-4o `o200k_base`, Qwen2/3, Gemma — 150K-260K vocabs deliberately widen multilingual + code coverage, cutting fertility on non-English/CJK at the cost of a huge unembed (handled with tied weights / z-loss).
- Llama 3 — jumped 32K -> 128K vocab vs Llama 2, improving compression ~15% and effective context.
- BLT / byte-latent (Meta, 2024) — matches BPE-Llama at scale with no tokenizer, fixing digit/spelling/rare-script brittleness.
- Multilingual + code models — fertility audits drive vocab allocation; digit tokenization (single-digit vs grouped) is set explicitly to help arithmetic.

## Common mistake

Comparing perplexity across models with different tokenizers, or assuming "more tokens" means "more information." A model with better compression has fewer tokens for the *same* text, so its per-token perplexity looks worse even when it is the same or better — normalize to bits-per-byte. Equally: treating vocab size as free. Bigger V helps compression but starves rare-token embeddings and inflates the softmax; the brittle failures (counting r's in "strawberry", arithmetic, exotic scripts) are baked in at tokenizer-training time and no amount of model scale fully removes them.

## See also
- [[bpe]] — the merge algorithm underlying BPE/BBPE tokenizers
- [[perplexity-bits-per-byte]] — the tokenizer-invariant metric for cross-model comparison
- [[softmax-bottleneck-logit-cap-final-layer-tying]] — large V makes the unembedding the dominant param/compute cost
