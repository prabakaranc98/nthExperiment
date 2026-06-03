# Dense Retrieval (Bi-Encoder Embeddings)

**One-liner:** Encode queries and documents *independently* into fixed-size vectors with a shared (or two-tower) encoder, then retrieve by nearest-neighbor similarity in embedding space — semantic matching beyond lexical overlap, and the retrieval half of RAG.

## The formula / definition

Two-tower (bi-encoder): score by a cheap similarity over pooled embeddings, with docs encoded *offline* and indexed.

```
q = pool(E_q(query))         # encode query  -> R^d
d = pool(E_d(doc))           # encode doc     -> R^d   (precomputed, indexed)
score(q, d) = <q, d>         # dot product, or cos(q,d) if L2-normalized
retrieve: top-k docs by score   # via ANN index (HNSW / IVF-PQ), not brute force
```

Trained contrastively with in-batch negatives (InfoNCE):

```
L = -log [ exp(s(q, d+)/τ) / ( exp(s(q,d+)/τ) + Σ_j exp(s(q, d-_j)/τ) ) ]
```

Key property: q and d never attend to each other → doc embeddings are query-independent → precompute once, search in O(log N) with ANN. This is the entire efficiency argument vs. a cross-encoder.

## Where it appears

- **DPR** (Karpukhin 2020) — the canonical BERT bi-encoder for open-domain QA; popularized in-batch negatives.
- **Contriever / E5 / GTE / BGE / Nomic / Jina v3** — unsupervised + contrastive pretraining at scale; the workhorse open embedding models of 2023-2025.
- **RAG** — first stage: embed the query, ANN-search the chunk index, feed top-k to the generator.
- **MTEB leaderboard** — the standard benchmark; 2024-2026 frontier is instruction-tuned LLM-based embedders (e5-mistral, NV-Embed, Qwen3-Embedding, Gemini/OpenAI `text-embedding-3`), often with Matryoshka dims.

## Common mistake

Treating the dense score as the final ranking. Bi-encoders trade accuracy for speed — they compress everything into one vector, losing fine-grained term interaction, so they recall well but rank imprecisely. The standard pipeline is dense (or hybrid) retrieval for top-k candidates, *then* a cross-encoder reranker. Also: cosine vs. dot product is not interchangeable — you must train and index with the same one, and most models require explicit L2 normalization.

## See also
- [[ann-vector-search]] — the index that makes top-k search sublinear
- [[infonce-contrastive-loss-with-temperature]] — the training objective
- [[cross-encoder-reranking]] — the precise second stage that fixes bi-encoder ranking
