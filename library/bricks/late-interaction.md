# Late Interaction (ColBERT / ColPali)

**One-liner:** Store *per-token* (ColBERT) or *per-patch* (ColPali) embeddings and score a query-document pair by summing the MaxSim — each query token's best match against any document token — instead of comparing two pooled vectors.

## The scoring function

Document → bag of L2-normalized token embeddings {d_1, ..., d_m}; query → {q_1, ..., q_n}. The relevance score is:

S(q, d) = Σ_{i=1}^{n} max_{j=1..m} ⟨q_i, d_j⟩

Each query term independently finds its single most similar document term (MaxSim), and those maxima are summed. Contrast with single-vector dense retrieval, which scores ⟨pool(q), pool(d)⟩ — one dot product on mean/CLS-pooled vectors. Late interaction defers the interaction to *after* encoding (hence "late"): no cross-attention between query and doc (that would be a cross-encoder / "all-interaction"), but matching is still fine-grained, not pooled.

## The serving recipe

1. **Offline:** encode every document once, store all token vectors in an index.
2. **Candidate gen:** ANN search over the flattened token vectors to retrieve docs whose tokens are near any query token.
3. **Rerank:** compute full MaxSim only on candidates.
ColBERTv2 adds residual/centroid compression (~quantize each token vector to a centroid ID + low-bit residual) to tame the m×dim storage blowup. PLAID optimizes the multi-stage candidate pruning.

## Where it appears

- **ColBERT / ColBERTv2 (Khattab & Zaharia 2020; Santhanam 2022)** — the original token-level late interaction for text retrieval; strong on out-of-domain (BEIR).
- **ColPali / ColQwen (2024)** — encode a document *screenshot* as ViT patch embeddings via a VLM, query as text tokens; MaxSim over patches retrieves visually-rich PDFs (tables, figures) with no OCR or layout parsing.
- **PLAID / ColBERT-Rerank** — used as a second-stage reranker over BM25 or dense candidates.

## Common mistake

Thinking it is the same cost as single-vector dense retrieval. Storage is per-token (m vectors per doc, often 100–1000×), and the index is far larger — late interaction buys accuracy and interpretable token-level matching at a real memory/latency price. Compression (ColBERTv2 residuals, binarization) is not optional at scale.

## See also
- [[dense-retrieval]] — the single-pooled-vector approach late interaction generalizes
- [[multimodal-visual-document-embeddings]] — ColPali's patch-level document-image setting
- [[cross-encoder-reranking]] — the full-interaction extreme; late interaction is the cheaper middle ground
