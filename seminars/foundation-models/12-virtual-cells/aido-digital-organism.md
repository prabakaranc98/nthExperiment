# AIDO — AI-Driven Digital Organism (GenBio AI)

*The most ambitious "digital organism" bet: a stack of integrated, multiscale foundation models meant to simulate biology from DNA up to tissue — `generative + biology + AI` as one connected system.*

**Why it's here:** AIDO is the concrete, public embodiment of the [virtual-cell](README.md) thesis taken to its limit — not one cell model, but a *connectable system* of foundation models across biological scales. It is also the real-world anchor for the virtual-cell stretch of the [Causal Foundation Model capstone](../../../capstones/causal-foundation-model/capstone.md): in AIDO's world a **perturbation is an intervention** — `do(gene = KO)` on single-cell expression — which is exactly the causal query a causal FM should answer.

**Primary sources:** [Vision paper (arXiv:2412.06993)](https://arxiv.org/abs/2412.06993) · [github.com/genbio-ai/AIDO](https://github.com/genbio-ai/AIDO) · [ModelGenerator](https://github.com/genbio-ai/ModelGenerator) · [huggingface.co/genbio-ai](https://huggingface.co/genbio-ai) · [genbio.ai](https://genbio.ai/)

---

## The thesis

> *"Toward an AI-Driven Digital Organism: multiscale foundation models for predicting, simulating, and programming biology at all levels."* — Song, Segal & Xing (2024)

The bet: biology is organized in a hierarchy (DNA → RNA → protein → structure → cell → tissue → organism), each scale has its own "language," and a **separate foundation model per scale** — pretrained self-supervised on that scale's data and then *connected* — can collectively act as a queryable, programmable in-silico organism. It is the foundation-model paradigm applied not to one biological modality but to the whole stack, with the modules designed to compose.

GenBio AI released **Phase 1** (Dec 2024): six model families spanning DNA, RNA, protein, protein structure, single-cell expression, and evolutionary information, plus the **ModelGenerator** adaptation library.

## The four-layer architecture

| Layer | Role |
|-------|------|
| **Data** | Multimodal biological corpora — genomes (796 species), transcriptomes, protein sequences/structures, ~50M single cells, perturbation screens |
| **Foundation models** | One self-supervised FM per scale (the AIDO.* modules below) — the reusable representations |
| **Downstream utility** | [ModelGenerator](https://github.com/genbio-ai/ModelGenerator): adapt / fine-tune / **fuse** modules for a task, with benchmark datasets + tutorials |
| **Applications** | Bioengineering & biomedicine — drug design, RNA/protein design, variant effects, perturbation response, synthetic biology |

## The model stack (Phase 1)

Module sizes and links from the [genbio-ai HF org](https://huggingface.co/genbio-ai) (~60 models, ~20 datasets).

| Module | Scale | Modality / job | Notes |
|--------|-------|----------------|-------|
| **AIDO.DNA** | 7B · 300M | Genomic language model (nucleotides) | Encoder-only; pretrained across **796 species**; variant-effect & regulatory tasks |
| **AIDO.RNA** | 1.6B (+ `-CDS`) | RNA language model | Among the largest RNA FMs; structure prediction, regulation, design |
| **AIDO.RNAIF** | 1.6B | RNA **inverse folding** (structure → sequence) | Design-oriented head on the RNA FM |
| **AIDO.Protein** | 16B (sparse MoE) | Protein language model | One of the largest protein LMs; MoE for compute efficiency |
| **AIDO.Protein-RAG** | 16B | Retrieval-augmented protein modeling | MSA/retrieval context (cf. ProteinGYM-DMS-RAG) |
| **AIDO.Protein2StructureToken** | 16B | Sequence → structure tokens | Bridges to the structure tokenizer |
| **AIDO.StructureTokenizer** | — | Discrete tokenizer for 3D protein structure | Makes structure a "language" the stack can model |
| **AIDO.Cell** | 100M · 10M · 3M (· 650M) | Single-cell expression FM | Pretrained on **~50M human cells**; first to ingest the **whole transcriptome** as input |
| **AIDO.Tissue** | 60M · 3M | Spatial / tissue-scale FM | Cell → tissue context |
| **genbio-pathfm** | ~1B | Pathology foundation model | Imaging scale |
| **STRING_GNN** | — | Protein–protein interaction GNN (STRING) | Network/evolutionary information |

**ModelGenerator** is the load-bearing piece for *using* this: a software package to rapidly adapt, fine-tune, and **fuse** pretrained AIDO modules on downstream tasks — the "downstream utility layer" made practical, with packaged benchmark datasets (perturbation, ProteinGYM-DMS-RAG, cell/RNA/protein/tissue downstream tasks, inverse folding).

## What transferred vs. what's bio-specific

**Transferred from language modeling:** the recipe is BERT/GPT-grade self-supervised pretraining (masked / autoregressive) on sequence data; the protein model even uses **mixture-of-experts** for efficient scaling — straight from the LLM playbook.

**Reinvented for biology:** the **multiscale composition** (separate FMs per scale, designed to connect), tokenizing 3D structure, ingesting the whole transcriptome per cell, and a **verifiable-ish reward** in places (does the designed RNA/protein fold/express?) — the property that makes biology an unusually good FM domain (see [06 · Protein & Biology](../06-protein-and-biology/README.md)).

## The connection to causal modeling (and the capstone)

This is the bridge worth dwelling on:

- A **gene knockout / overexpression / drug** in a Perturb-seq screen is literally an **intervention** `do(gene = state)` on the cell's expression distribution. Predicting the post-perturbation transcriptome is **interventional prediction** — `P(expression | do(gene = KO))`.
- AIDO.Cell + the `foundation-models-perturbation` dataset are an expression-scale FM and an interventional benchmark. GEARS (Nat. Biotech 2023) is the standard perturbation baseline; the 2025 *Nature Methods* benchmark cautions that deep models still struggle to beat linear baselines — a live, falsifiable target.
- The [Causal Foundation Model capstone](../../../capstones/causal-foundation-model/capstone.md) proposes a PFN-style causal FM over *synthetic* SCMs as the spine, with **virtual-cell perturbation prediction as the killer stretch** — and AIDO.Cell is exactly the backbone/benchmark to test that against. *Causal FM (the method) × AIDO/virtual-cell (the domain) = the research thread.*

## Open questions / what to watch

- **Does multiscale integration actually compose?** Phase 1 shipped strong *per-scale* FMs; the harder, unproven claim is that DNA→RNA→protein→cell modules *connect* into a coherent organism-level simulator.
- **Perturbation prediction is unsolved.** Beating simple baselines on held-out (unseen-gene) perturbations remains contested — the honest benchmark for any "virtual cell."
- **Causal vs. correlational.** These are predictive FMs; whether they encode *interventional* structure (vs. fitting `P(Y|X)`) is exactly the question a causal FM is built to probe — see [why-causality](../../../library/causal-ml/why-causality.md) and [do-calculus](../../../library/bricks/do-calculus.md).

## See also

- [12 · Virtual Cells](README.md) — the domain this anchors
- [06 · Protein & Biology](../06-protein-and-biology/README.md) · [07 · Omics & Genomics](../07-omics-and-genomics/README.md) — adjacent scales
- [Causal Foundation Model capstone](../../../capstones/causal-foundation-model/capstone.md) — perturbation-as-intervention stretch
- [APSL — Algorithmic Probabilistic Structure Learning](../../apsl-structure-learning.md) · [Data Foundations](../../data-foundations.md)
