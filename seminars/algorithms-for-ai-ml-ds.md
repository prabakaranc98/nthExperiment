# Algorithms for AI / ML / DS

*The algorithmic toolkit behind modern AI/ML/DS: when the data is too big to touch twice, you **randomize, sketch, sample, stream, or exploit structure** — and trade an exact answer for a provably-good approximate one, fast.*

**FAIRE context:** the algorithmic spine of [505 Systems](../faire-program.md) and [506 Applied Math](../faire-program.md); the build-it-yourself companion to the [09 · Algorithms & Systems](frontier-ai/09-algorithms-and-systems/README.md) reading topic. Read alongside [Data Foundations](data-foundations.md) (where MinHash/SemDeDup/streaming run at trillion-token scale) and the [bricks](../library/bricks/README.md).

---

## The one idea

Classical algorithms ask "what's the exact answer?" At AI/ML/DS scale (`n` in the billions, `d` in the thousands, data arriving as a stream you see once) that's the wrong question. These methods ask instead: **"what's a provably-good approximate answer I can get in sublinear time / one pass / low memory, with high probability?"** The currencies are *randomness* (sample, project, hash), *structure* (graphs, low rank, sparsity), and *concentration* (why a random estimate is reliable).

| Cluster | What you get | Where it shows up |
|---|---|---|
| [Concentration & probabilistic method](#1-concentration--the-probabilistic-method) | why random estimates are trustworthy | generalization bounds, eval CIs, scaling-law fits |
| [Randomized numerical linear algebra](#2-randomized-numerical-linear-algebra-randnla) | fast low-rank, regression, PCA | LoRA, PCA, Performer/random-features attention |
| [Dimensionality reduction & projections](#3-dimensionality-reduction--random-projections) | shrink `d`, keep geometry | embeddings, feature hashing, fast kNN |
| [Similarity search & ANN](#4-similarity-search--ann) | nearest neighbors in sublinear time | vector DBs / RAG, dedup, near-dup detection |
| [Streaming & sketching](#5-streaming--sketching) | one-pass, low-memory stats | data pipelines, online metrics, counting at scale |
| [Graph algorithms & graph learning](#6-graph-algorithms--graph-based-learning) | structure in relations | clustering, ranking, GNNs, knowledge graphs |
| [Network flows, matching & combinatorial opt](#7-network-flows-matching--combinatorial-optimization) | optimal assignment & cuts | DETR matching, segmentation, scheduling |
| [Optimal transport](#8-optimal-transport-the-continuous-flow) | distance between distributions | Wasserstein, Sinkhorn, **flow matching** |
| [Probabilistic data structures & hashing](#9-probabilistic-data-structures--hashing) | approximate membership/counting | dedup, caching, cardinality, frequency |
| [Algorithm-design paradigms](#10-algorithm-design-paradigms-the-backbone) | the underlying moves | DP (Viterbi/beam), D&C, approximation, online |

---

## 1. Concentration & the probabilistic method

The foundation: *why* a random sample, sketch, or estimate is close to the truth with high probability. Every guarantee below cashes out to one of these inequalities.

| Topic | Key results | Why it matters |
|---|---|---|
| Tail bounds | Markov → Chebyshev → **Chernoff / Hoeffding** → Bernstein (variance-aware) | sample-complexity & how many samples to trust an estimate to ±ε |
| Martingale bounds | Azuma–Hoeffding, **McDiarmid** (bounded differences) | generalization bounds, stability, online regret |
| Matrix concentration | matrix Chernoff / **Matrix Bernstein** (Tropp) | the engine behind RandNLA error bounds |
| The probabilistic method | union bound, first/second-moment, Lovász Local Lemma | existence proofs; "a random construction works whp" |
| Las Vegas vs Monte Carlo | always-correct-variable-time vs fixed-time-maybe-wrong | the two flavors of every randomized algorithm |

*Ties to:* eval statistics & CIs ([501](../faire-program.md)), PAC/PAC-Bayes ([bias-variance](../library/bricks/bias-variance.md), [pac-bayes](../library/bricks/pac-bayes.md)).

## 2. Randomized numerical linear algebra (RandNLA)

Exact SVD/least-squares is `O(nd·min(n,d))` — too slow at scale. Sketch the matrix down first, then solve the small problem. *(Your "Randomized SVD I/II", "Matrix sketching", and "Randomized regression" weeks live here.)*

| Topic | Key methods | Why it matters |
|---|---|---|
| Randomized SVD / low-rank | random projection → range-finder → small SVD (Halko–Martinsson–Tropp) | fast PCA, low-rank approximation, the math under [SVD](../library/bricks/svd.md)/[LoRA](../library/bricks/lora.md) |
| Sketching | **JL / Gaussian sketch**, CountSketch, SRHT (subsampled Hadamard) | compress `n×d` → `s×d` preserving the action of the matrix |
| Leverage scores | importance-sample rows by leverage | which rows actually matter for a least-squares fit |
| Randomized regression | sketch-and-solve, iterative sketching (Blendenpik / LSRN) | least-squares on huge `n` in near-linear time |
| Kernel / attention approx | **random features** (Rahimi–Recht), Nyström, CUR | Performer/linear-attention, scalable kernel methods |

## 3. Dimensionality reduction & random projections

Shrink `d` while preserving the geometry that downstream learning needs.

| Topic | Key results | Why it matters |
|---|---|---|
| **Johnson–Lindenstrauss** | `n` points → `O(ε⁻²log n)` dims, distances preserved ±ε | the theorem that makes random projection *work*; ANN, sketching |
| Random projection | Gaussian / sparse (Achlioptas) / fast-JL | cheap dimensionality reduction, feature compression |
| PCA family | exact, **randomized PCA**, incremental/streaming PCA | the workhorse linear reducer; ties to [eigendecomposition](../library/bricks/eigendecomposition.md) |
| Nonlinear / manifold | t-SNE, UMAP (neighbor-graph + force layout) | visualization, structure discovery (use with care — see the t-SNE caveats) |
| Feature hashing | the "hashing trick" | huge sparse feature spaces in fixed memory |

## 4. Similarity search & ANN

Exact nearest-neighbor is linear per query; ANN gets it sublinear. *(Your "ANN I/II" weeks.)* The backbone of vector databases, RAG, and dedup.

| Topic | Key methods | Why it matters |
|---|---|---|
| **LSH** | hyperplane (cosine), p-stable (ℓ₂), the LSH framework | sublinear similarity search with collision-probability guarantees |
| Set / text similarity | **MinHash** (Jaccard), SimHash (cosine), shingling | near-duplicate detection → **data dedup** ([Data Foundations](data-foundations.md)) |
| Quantization | Product Quantization (PQ), OPQ, scalar/residual quant | compress vectors 8–32× for billion-scale indexes |
| Graph-based ANN | **HNSW**, NSG, DiskANN | the SOTA recall/latency frontier — what vector DBs ship |
| Inverted indexes | IVF, IVF-PQ, ScaNN | coarse-quantize then search a shortlist; the FAISS toolkit |

*Ties to:* RAG retrieval, [conformal](../library/bricks/conformal.md)-style selective retrieval, embedding search.

## 5. Streaming & sketching

You see the data **once**, in order, with memory ≪ `n`. *(Your "Streaming I/II" weeks.)* Exactly the regime of a trillion-token data pipeline.

| Topic | Key methods | Why it matters |
|---|---|---|
| Frequency / heavy hitters | **Count-Min Sketch**, Count-Sketch, Misra–Gries, Space-Saving | top-k tokens/keys in one pass, tiny memory |
| Frequency moments | **AMS sketch** (F₂ / distinct-ish) | the canonical streaming lower-bound/upper-bound result |
| Cardinality | **HyperLogLog**, Flajolet–Martin | count distinct items (uniques, vocab) in ~KB |
| Sampling from a stream | **reservoir sampling**, weighted/distinct sampling | uniform sample of an unbounded stream (training data subsampling) |
| Streaming linear algebra | **Frequent Directions**, streaming/online PCA | low-rank summaries of a matrix you can't store |
| Graph streams | semi-streaming connectivity, sparsifiers, triangle counting | graph stats when the edge list is a stream |

## 6. Graph algorithms & graph-based learning

Relations are everywhere (citation, social, knowledge, attention). *(Your "Graph-based learning I/II" weeks — expanded.)*

| Topic | Key methods | Why it matters |
|---|---|---|
| Traversal & paths | BFS/DFS, **Dijkstra**, A*, Bellman–Ford, Floyd–Warshall | the primitives; A* underlies many search/planning systems |
| Trees & connectivity | MST (Kruskal/Prim), union-find, SCCs (Tarjan) | clustering, connectivity, dependency analysis |
| **Spectral graph theory** | Laplacian, Fiedler vector, Cheeger | spectral clustering, graph partitioning, the math under GNNs |
| Clustering / community | **spectral clustering**, modularity, **Louvain**/Leiden | community detection, segmentation |
| Ranking & walks | **PageRank**, HITS, personalized PageRank, random walks | retrieval ranking, influence, the original Google algorithm |
| Node embeddings | **DeepWalk / node2vec**, LINE; label propagation | self-supervised graph features → feeds GNNs |
| Graph partitioning | METIS, balanced cuts | distributing graphs/models across machines |

*Ties to:* GNNs (DL track), knowledge graphs, [GFM-RAG](data-foundations.md)-style graph retrieval.

## 7. Network flows, matching & combinatorial optimization

The "flows" family you asked for — optimal assignment, cuts, and matchings, which show up far more in ML than people expect.

| Topic | Key methods | Why it matters |
|---|---|---|
| **Max-flow / min-cut** | Ford–Fulkerson, Edmonds–Karp, **Dinic**, push–relabel | image segmentation (graph cuts), reliability, min-cut clustering |
| Min-cost flow | cycle-canceling, SSP, network simplex | transportation/assignment at scale |
| Bipartite matching | **Hungarian** algorithm, Hopcroft–Karp, auction | **DETR / set-prediction** object detection, label assignment |
| LP & duality | simplex, interior-point; LP relaxation + rounding | the lens unifying flows, matching, and approximation |
| Submodular optimization | greedy (1−1/e), lazy-greedy | data selection, sensor placement, coreset/active learning |
| Approximation | LP-rounding, primal–dual, local search | NP-hard problems with provable ratios |

## 8. Optimal transport (the continuous "flow")

Where combinatorial flow meets probability — moving mass between distributions. A direct bridge to modern generative modeling.

| Topic | Key methods | Why it matters |
|---|---|---|
| Monge–Kantorovich | the OT problem; **Wasserstein** distance | a geometry-aware distance between distributions |
| Entropic OT | **Sinkhorn** iterations (fast, differentiable) | WGAN, domain adaptation, differentiable matching |
| Dynamic OT | Benamou–Brenier; velocity fields | the continuity equation behind **[flow matching](../library/bricks/flow-matching.md)** & [diffusion](../library/bricks/ddpm.md) |

## 9. Probabilistic data structures & hashing

Approximate membership, counting, and dedup in tiny memory. *(MinHash/SimHash also live in §4; the filters live here.)*

| Topic | Key methods | Why it matters |
|---|---|---|
| Membership | **Bloom filter**, counting Bloom, **Cuckoo** / Quotient filter | "have I seen this?" — dedup, caching, crawl frontiers |
| Hashing | universal/perfect hashing, **consistent hashing**, FNV/xxHash | hash tables, sharding, distributed caches |
| Sampling estimators | importance sampling, **MCMC / Gibbs**, rejection sampling | estimate intractable expectations (RL gradients, Bayesian inference) |
| Cardinality/frequency | HyperLogLog, Count-Min (cross-ref §5) | analytics over massive key spaces |

## 10. Algorithm-design paradigms (the backbone)

The moves underneath everything above — worth being fluent in for their own sake.

| Paradigm | Canonical AI/ML instances |
|---|---|
| **Dynamic programming** | Viterbi, CYK parsing, **beam search**, edit distance, RL value/policy iteration & Bellman |
| Divide & conquer | FFT, merge/quick-select, fast matrix multiply (Strassen) |
| Greedy | Huffman/BPE merges, submodular maximization, MST |
| Randomization | everything in §1–§5; QuickSort, primality, hashing |
| Approximation & online | LP-rounding; **bandits/regret**, online convex optimization, competitive analysis |
| Sublinear / property testing | estimate a global property from `o(n)` samples |

---

## How to study this

Build, don't just read — implement one algorithm per cluster from scratch and *measure* the approximation/runtime tradeoff (the whole point is the tradeoff, so prove it empirically with error bars — the [501](../faire-program.md) rigor applies here too).

1. **Concentration first** (§1) — it's the *why* for everything else; without it the rest is recipes.
2. **RandNLA + JL + ANN** (§2–4) — the randomized/geometry core; one project: randomized SVD, and an LSH/HNSW index you benchmark against brute force.
3. **Streaming** (§5) — implement Count-Min + reservoir sampling + HyperLogLog on a real stream; this is the data-pipeline skill.
4. **Graphs + flows + OT** (§6–8) — spectral clustering, a max-flow min-cut segmenter, and a Sinkhorn solver (then connect Sinkhorn → flow matching).
5. **Paradigms** (§10) as the connective tissue — name the paradigm behind each method you meet.

**Highest-leverage texts:** **Blum, Hopcroft & Kannan — [*Foundations of Data Science*](https://home.ttic.edu/~avrim/book.pdf)** (📖 the spine of this whole map); Mitzenmacher & Upfal (*Probability and Computing*); Woodruff ([*Sketching for NLA*](https://arxiv.org/abs/1411.4357)); Spielman (*Spectral Graph Theory*); Peyré & Cuturi ([*Computational Optimal Transport*](https://arxiv.org/abs/1803.00567)); CLRS for the paradigms. → full list in [The Canon](../library/reads-and-references/00-books-the-canon.md).

## See also

- [Data Foundations](data-foundations.md) — MinHash/SemDeDup, streaming dedup, and sketching at trillion-token scale
- [Statistical & Probabilistic Foundations](statistical-probabilistic-foundations.md) — the concentration/probability backing
- [09 · Algorithms & Systems](frontier-ai/09-algorithms-and-systems/README.md) — the paper-reading topic this builds out
- [Concept library (bricks)](../library/bricks/README.md) — svd · flow-matching · roofline · conformal
