# Learned Sparse Retrieval (SPLADE)

**One-liner:** Use a transformer's MLM head to predict a sparse, term-weighted vocabulary vector (with learned term expansion) for queries and documents, so retrieval runs on a standard inverted index but with learned semantics — the sparse leg of hybrid search.

## The formula / definition

For each input token i, project its contextual embedding through the MLM head onto the vocab (|V| logits w_ij per vocab term j). Aggregate over the sequence with log-saturation and max-pooling:

w_j = max over i of  log(1 + ReLU(w_ij))

This gives one nonnegative weight per vocab term → a |V|-dim vector (~30k for BERT) that is mostly zero. ReLU + max-pool drives sparsity; the log dampens dominant terms. Term *expansion* falls out for free: a document about "dog" gets nonzero weight on "puppy", "canine", "pet" via the MLM head — no exact lexical match required.

Score = dot product of query and doc sparse vectors, computed on an inverted index (sum over the few shared nonzero terms).

**Sparsity is trained, not post-hoc:** add an L1 (originally) or FLOPS regularizer on the weights to the ranking loss (contrastive / distillation from a cross-encoder). The FLOPS term penalizes expected query–doc term overlap cost, keeping the index cheap to serve.

L = L_rank(contrastive, often with distillation) + λ_q·FLOPS(q) + λ_d·FLOPS(d)

## Where it appears

- **SPLADE / SPLADEv2 / SPLADE++ (Formal et al., 2021–2022)** — the canonical learned sparse retriever; SPLADE++ adds cross-encoder distillation + hard negatives to close the gap with dense retrieval on BEIR
- **Hybrid search (2024–2026 RAG stacks)** — SPLADE/uniCOIL as the sparse signal fused with a dense embedder via reciprocal rank fusion; strong on out-of-domain and rare-term / exact-match queries where dense models drift
- **Elasticsearch ELSER & OpenSearch neural sparse** — productionized learned-sparse models served on Lucene impact-ordered inverted indexes; a default "neural without ANN" option
- **uniCOIL / DeepImpact / TILDE** — related learned term-weighting schemes (no or limited expansion vs. SPLADE's full vocab expansion)

## Common mistake

Treating SPLADE as "just BM25 with neural weights." The defining feature is **term expansion** — it adds vocab terms that never appear in the text — so it matches semantically like a dense model while staying inverted-index-native. Also: skipping the FLOPS/sparsity regularizer. Without it the vectors densify, query latency explodes, and you lose the entire reason to use an inverted index over an ANN dense index.

## See also
- [[dense-retrieval]] — the dense counterpart; SPLADE is the sparse leg fused with it
- [[hybrid-search-reciprocal-rank-fusion]] — how sparse + dense scores are combined in practice
- [[cross-encoder-reranking]] — the teacher signal distilled into SPLADE++ and the second-stage reranker over its candidates
