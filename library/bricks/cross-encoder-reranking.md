# Cross-Encoder Reranking

**One-liner:** Feed query and candidate together through one transformer with full cross-token attention to emit a single relevance score, then resort the first-stage shortlist — the standard precision-boosting second stage of retrieval.

## The formula / definition

Bi-encoder (first stage): score = sim(E(q), E(d)), embeddings precomputed and indexed → cheap, recallable via ANN.

Cross-encoder (rerank): score = w·CLS(Transformer([q] ++ [SEP] ++ [d])), a scalar from a head on top of joint encoding. Every query token attends to every document token, so there is no precomputable document vector — you must run a forward pass *per (query, candidate) pair*.

```
shortlist = bi_encoder_retrieve(q, k=100)      # cheap, high recall
scores    = [cross_encoder(q, d) for d in shortlist]  # k forward passes
return topn(sort(shortlist, by=scores), n=10)  # high precision
```

Cost: O(k) transformer passes at query time — the reason it is a reranker over a small k, not a retriever over millions.

## Where it appears

- **MS MARCO / monoBERT** (Nogueira & Cho, 2019) — the canonical BERT cross-encoder reranker; pairwise/listwise variants (monoT5, duoT5) followed.
- **RAG pipelines** — Cohere Rerank, BGE-reranker, mxbai-rerank, Jina reranker sit between vector search and the LLM context window to cut irrelevant chunks.
- **LLM rerankers (2024-2026)** — RankGPT / RankZephyr / RankLLaMA do listwise reranking by prompting an LLM to order passages; distilled into smaller cross-encoders for latency.

## Common mistake

Trying to use a cross-encoder as the first-stage retriever. You cannot index it — scoring is joint, so there is no document embedding to put in an ANN index. It only reranks a candidate set produced by a cheaper retriever (dense, BM25, or hybrid). Conversely, skipping reranking and trusting bi-encoder cosine scores alone leaves a large precision gap.

## See also
- [[dense-retrieval]] — the bi-encoder first stage whose shortlist the cross-encoder reranks
- [[late-interaction]] — ColBERT's middle ground: token-level scoring but precomputable, cheaper than a full cross-encoder
- [[hybrid-search-reciprocal-rank-fusion]] — common way to build the candidate set fed into reranking
