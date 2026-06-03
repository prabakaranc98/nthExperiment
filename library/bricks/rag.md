# RAG (Retrieval-Augmented Generation)

**One-liner:** Condition generation on passages fetched at inference from an external corpus, so answers draw on non-parametric, updatable knowledge rather than only weights — the default knowledge-intensive architecture.

## The formula / pseudocode

Retrieve-then-read pipeline:

1. Embed query: q = E_q(x)
2. Retrieve top-k passages from index: D_k = top-k_{d∈corpus} sim(q, E_d(d))  (sim = dot product / cosine, via ANN)
3. Generate conditioned on retrieved context: p(y | x) = LLM(y | x, d_1, ..., d_k)

The original RAG (Lewis et al., 2020) marginalizes over passages:

  p(y | x) ≈ Σ_{d ∈ D_k} p_retriever(d | x) · p_gen(y | x, d)

Two variants: **RAG-Sequence** (one d for the whole output) vs **RAG-Token** (re-marginalize per token). Modern systems skip marginalization: concatenate the k passages into the prompt and decode once. Retriever and generator are usually trained/served separately; the index is rebuilt, not the weights.

## Where it appears

- **Lewis et al. (2020)** — original RAG: DPR retriever + BART generator, end-to-end with retriever gradients.
- **Atlas / RETRO / REALM** — retrieval baked into pretraining; RETRO uses chunked cross-attention over a trillion-token datastore.
- **Production stacks (2024-2026)** — query rewriting → hybrid dense+BM25 retrieval → cross-encoder rerank → long-context LLM read; the standard enterprise/grounded-QA pattern.
- **Agentic RAG** — the LLM issues retrieval as a tool call in a loop (decide what/when to fetch) rather than one fixed retrieve step.

## Common mistake

Treating retrieval quality as solved once recall is high, then blaming the generator for wrong answers. Most RAG failures are upstream: chunking that splits the answer, no reranking so the right passage sits at rank 30, or the LLM ignoring/over-trusting a retrieved-but-irrelevant passage. Recall@k of the retriever and faithfulness of the reader are separate axes — measure and fix them separately.

## See also
- [[dense-retrieval]] — the retriever that produces the candidate passages
- [[cross-encoder-reranking]] — reorders retrieved candidates before the generator reads them
- [[agentic-iterative-retrieval]] — RAG as a multi-step tool-using loop instead of one fetch
