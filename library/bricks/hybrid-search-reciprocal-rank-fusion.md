# Hybrid Search & Reciprocal Rank Fusion

**One-liner:** Run dense (semantic embedding) and sparse/lexical (BM25, SPLADE) retrieval in parallel, then fuse their ranked lists with Reciprocal Rank Fusion to capture both meaning and exact-term/rare-token matches — the pragmatic production-RAG default.

## The formula

RRF scores a document by summing, over each retriever, the reciprocal of its rank in that retriever's list:

RRF(d) = Σ_r 1 / (k + rank_r(d))

- rank_r(d) = 1-based position of d in retriever r's results (skip if not retrieved by r)
- k = smoothing constant, conventionally **60** (Cormack et al., 2009). Larger k flattens the curve, reducing the dominance of top ranks.

Key property: **rank-based, not score-based** — needs no calibration across retrievers whose scores live on incompatible scales (cosine similarity ∈ [−1,1] vs. unbounded BM25 tf-idf sums). That scale-invariance is why RRF beats naive score addition.

## Where it appears

- **Production RAG stacks** — Elasticsearch/OpenSearch `rrf` retriever, Weaviate/Qdrant/Pinecone/Milvus hybrid endpoints, pgvector + ts_rank; default when neither modality alone suffices.
- **BEIR / MTEB findings** — dense models win on semantic/paraphrase queries but lose to BM25 on out-of-domain, entity-heavy, and rare-token (codes, IDs, jargon) queries; hybrid is robust across the distribution.
- **SPLADE / learned sparse** — sparse-but-learned term weights fuse cleanly with dense vectors; both are "neural," but the lexical side still anchors exact matches.
- **Two-stage pipelines** — RRF as cheap first-stage fusion feeding a cross-encoder reranker for the final top-k.

## Common mistake

Treating RRF as a quality-aware blend. It ignores the *magnitude* of relevance scores and absolute score gaps — a doc ranked #1 by a confident retriever and #1 by a near-random one contribute identically. When one retriever is clearly stronger, weighted RRF (per-retriever weights or tuned k) or learned fusion outperforms vanilla RRF; uniform RRF is a strong, tuning-free baseline, not the ceiling.

## See also
- [[rag]] — hybrid retrieval is the standard front-end for retrieval-augmented generation
- [[dense-retrieval]] — the semantic half of the fusion
- [[cross-encoder-reranking]] — the precision reranker that typically consumes RRF output
