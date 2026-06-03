# Multimodal / Visual Document Embeddings

**One-liner:** Embed rendered document pages directly as images (or jointly text+image) with a vision-language model so layout, tables, and figures are retrievable end-to-end — skipping the brittle OCR → layout-parse → chunk pipeline that dominates enterprise PDF RAG.

## The key insight

Don't extract text and embed it. Render the page to a pixel image, run it through a VLM encoder, and index the resulting patch-level embeddings. Retrieval is a **late-interaction** match between query token vectors and page patch vectors (ColBERT-style MaxSim):

    score(q, d) = Σ_{i∈query toks} max_{j∈page patches} ⟨ E_q[i], E_d[j] ⟩

Each page → a *multi-vector* tensor of shape (num_patches × dim), e.g. ~1024 patches × 128-d. The max-over-patches lets a query token "find" the one figure caption / table cell that answers it, without OCR ever segmenting the page. (A single-vector variant just mean-pools to one embedding per page for cheaper ANN, trading recall for storage.)

## Where it appears

- **ColPali / ColQwen2 (Faysse et al., 2024–2025)** — PaliGemma / Qwen2-VL vision encoder + ColBERT late interaction; the reference architecture, evaluated on the **ViDoRe** benchmark (visual document retrieval). Beats OCR+text pipelines on table/figure-heavy docs.
- **DSE / VisRAG / ColNomic, voyage-multimodal-3, Cohere Embed v3/v4 multimodal** — production multimodal embedding endpoints that take page images directly; some pool to a single dense vector for standard ANN indexes.
- **Enterprise PDF RAG (2025–2026)** — replaces Unstructured/layout-parser chunkers; the retrieved *page image* is then fed straight to a VLM generator (Qwen2-VL, GPT-4o, Gemini) for grounded answering.

## Common mistake

Treating it as a free win and forgetting the **storage / cost blowup**: multi-vector page embeddings are ~100–1000× the footprint of one dense vector per chunk, and MaxSim isn't a plain ANN lookup — you need binarization, pooling, or a two-stage (cheap retrieve → late-interaction rerank) setup. Also: it doesn't kill OCR for *everything* — pure long-form prose still retrieves fine and far cheaper with text chunks; the win is concentrated in visually-structured pages.

## See also
- [[late-interaction]] — the ColBERT MaxSim scoring that makes patch-level page matching work
- [[clip-contrastive-vision-language-pretraining]] — the vision-language alignment these encoders are built on
- [[rag]] — the retrieval pipeline this reshapes for document-heavy corpora
