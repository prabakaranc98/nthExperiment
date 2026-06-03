# Byte-Pair Encoding (BPE)

**One-liner:** A greedy tokenizer-training algorithm that starts from bytes/characters and iteratively merges the most frequent adjacent symbol pair into a new token, building a subword vocabulary that trades sequence length against vocab size.

## The algorithm

Training (learn the merge table):
```
1. Initialize vocab = set of base symbols (bytes 0-255 for byte-level BPE)
2. Pre-tokenize corpus into words; represent each as a sequence of symbols
3. Repeat until |vocab| == target_size:
     a. count freq of every adjacent symbol pair across the corpus
     b. (A, B) = argmax pair frequency
     c. add merged token "AB" to vocab; record merge rule (A,B)->AB
     d. replace all adjacent (A,B) with AB in the corpus
```
Encoding: apply the learned merges to new text **in the order they were learned** (rank-priority), greedily and deterministically. The merge list, not a probability model, defines the tokenizer.

Cost: vocab size V is the single knob. Larger V -> shorter sequences (fewer tokens, cheaper O(N²) attention) but a bigger embedding/softmax matrix and more rare tokens. Typical frontier V ≈ 100k-256k.

## Where it appears

- GPT-2/3/4, Llama, Mistral — **byte-level BPE**: merges operate over raw UTF-8 bytes, so every string is encodable with zero OOV and no `<unk>`.
- GPT-4 / GPT-4o (`tiktoken`, `cl100k_base`/`o200k_base`) — BPE with ~100k-200k vocab; o200k widened multilingual + code coverage.
- SentencePiece — implements BPE (and unigram-LM) directly on raw text with `▁` marking spaces; used by Llama, T5, Gemma.
- WordPiece (BERT) — a close cousin: merges the pair maximizing corpus likelihood gain, not raw frequency.

## Common mistake

Thinking BPE tokenizes by probability or "meaning." It is a fixed, deterministic merge table applied greedily by rank — no model, no context. This is why arithmetic, spelling, and rare-word tasks break (the famous "strawberry" / digit-grouping failures): the segmentation is frozen at tokenizer-training time and is blind to the downstream task. Also: token counts are not character counts, and merges are corpus-specific.

## See also
- [[softmax]] — vocab size V sets the output softmax/embedding dimension and its cost
- [[scaling-laws]] — tokenization changes the effective token count D and shifts the loss curve
- [[in-context-learning]] — tokenization artifacts (digit/space splits) bound what the model can do in-context
