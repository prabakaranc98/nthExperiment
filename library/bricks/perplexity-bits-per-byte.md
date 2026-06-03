# Perplexity & Bits-Per-Byte

**One-liner:** Perplexity is exp of per-token cross-entropy (the effective branching factor of the next-token distribution); bits-per-byte renormalizes the same loss to raw UTF-8 bytes so models with different tokenizers can be compared on the same footing.

## The formula / definition

Per-token cross-entropy over a sequence (nats): L = −(1/T) Σₜ log p(xₜ | x_<t)

Perplexity: **PPL = exp(L)** = exp(−(1/T) Σₜ log p(xₜ | x_<t))

Bits-per-token: BPT = L / ln 2 (just CE in base-2). Bits-per-byte rescales by the token-to-byte ratio:

**BPB = (1/ln 2) · (Σₜ −log p(xₜ | x_<t)) / N_bytes = BPT · (N_tokens / N_bytes)**

where N_bytes is the UTF-8 byte length of the *same text*. Bits-per-character (BPC) is the analogous per-character form. Total log-prob of the text is tokenizer-invariant (it's a product over the same string); dividing by bytes/chars removes the tokenizer's influence, dividing by tokens does not.

## Where it appears

- **GPT-2/3 reports** — PPL on WikiText/PTB/LAMBADA; the canonical intrinsic LM number before benchmark-suites took over
- **The Pile / GPT-3 paper** — popularized **bits-per-byte** precisely so models with different vocabularies are comparable
- **Pythia, LLM eval harnesses (EleutherAI lm-eval)** — `byte_perplexity` / `bits_per_byte` as standard tokenizer-agnostic metrics
- **Scaling laws (Kaplan, Chinchilla)** — the loss L whose power-law decay is fitted is this per-token CE; PPL = exp(L)
- **Tokenizer / data-mixture ablations (2024-2026)** — BPB compares a BPE model vs a byte/char model, or different vocab sizes, on equal terms

## Common mistake

Comparing raw perplexity across models with different tokenizers. A coarser tokenizer packs more text per token, so it has fewer tokens per byte and a *lower* per-token PPL for the same actual compression — the number looks better without the model being better. Always convert to bits-per-byte (or per-character) for cross-tokenizer claims. Also: PPL is only meaningful per-token within a fixed vocabulary, and depends on context length / sliding-window stride used to score long documents.

## See also
- [[cross-entropy]] — perplexity is literally exp of the CE loss; BPB is CE in base-2 renormalized to bytes
- [[bpe]] — the tokenizer choice that makes raw PPL incomparable and motivates bits-per-byte
- [[scaling-laws]] — the loss whose predictable decay underlies all scaling-law fits
