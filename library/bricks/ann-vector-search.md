# ANN Vector Search (HNSW / IVF-PQ)

**One-liner:** Approximate nearest-neighbor indexes that trade exactness for sublinear query time via navigable small-world graphs (HNSW) or coarse-cluster routing plus product-quantized residuals (IVF-PQ) — the retrieval infrastructure under every RAG and dense-retrieval system.

## The two dominant index families

**HNSW (graph-based, Malkov & Yashunin 2018).** Build a multi-layer proximity graph; upper layers are sparse (long-range "express lanes"), the bottom layer holds all points. Query = greedy best-first search from the top entry point, descending layers, keeping a candidate heap of size `efSearch`.
- Knobs: `M` (edges/node), `efConstruction` (build quality), `efSearch` (recall↔latency at query time).
- Build ~O(N log N), query ~O(log N), memory ~O(N·M) — graph lives in RAM, no compression by default. Highest recall/latency, worst memory.

**IVF-PQ (quantization-based, Jégou et al. 2011).** Two-stage:
1. **IVF (inverted file):** k-means into `nlist` Voronoi cells; query probes the `nprobe` nearest cells (coarse routing, prunes the search).
2. **PQ:** split each D-dim residual vector into `m` subvectors, quantize each to one of 256 centroids → store as `m` bytes. Distances computed via precomputed lookup tables (ADC — asymmetric distance computation), summing per-subvector table hits.

PQ compression: 768-dim fp32 (3072 B) → m=64 codes (64 B) ≈ 48x smaller. Recall↔compression set by `m` and bits/code; recall↔speed by `nprobe`.

## Where it appears

- **RAG / dense retrieval** — FAISS, ScaNN, hnswlib, and vector DBs (Milvus, Qdrant, Weaviate, pgvector) back the retrieval step; HNSW is the default for high-recall low-latency, IVF-PQ for billion-scale on bounded RAM.
- **DiskANN / Vamana (Microsoft)** — graph index that spills to SSD with PQ-compressed vectors in RAM, enabling billion-scale single-machine ANN.
- **ScaNN (Google)** — anisotropic PQ that weights quantization error along the query-relevant (inner-product) direction; strong MIPS recall.

## Common mistake

Treating PQ-reconstructed distances as exact, then ranking final results on them. PQ distances are lossy approximations for *candidate generation*; production systems **rerank** the shortlist with full-precision vectors (or a cross-encoder). Also: building an index for L2 when your embeddings are trained for cosine/inner-product — normalize first, or recall silently collapses.

## See also
- [[dense-retrieval]] — produces the embeddings these indexes search over
- [[rag]] — the consuming system; ANN is its retrieval substrate
- [[cross-encoder-reranking]] — the precise second stage that fixes ANN's approximate ranking
