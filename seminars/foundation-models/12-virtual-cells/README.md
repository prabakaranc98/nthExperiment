# 12 · Virtual Cells Foundation Models

*Learn the cell's response function: given a perturbation, predict the new state.*

## The adaptation
- **Token** — a gene, represented by its expression level in a cell. A cell is a "sentence" of ranked or value-encoded genes; rank-based (Geneformer) and value-based (scGPT, scFoundation) encodings dominate.
- **Pretraining objective** — masked gene-expression prediction over tens to hundreds of millions of cells (self-supervised). Frontier perturbation models add a supervised state-transition objective: predict the post-perturbation transcriptome.
- **Inductive bias** — cells are points in a learned state manifold; perturbations are operators that move them. The "central dogma" structure (gene → expression → phenotype) is the prior, not free-form sequence.
- **Verification signal** — wet-lab ground truth: CRISPR/Perturb-seq screens, drug-response assays. Held-out perturbations (unseen genes, unseen cell contexts) are the real test, not reconstruction loss.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| scGen | Lotfollahi et al. | 2019 | [Nature Methods](https://www.nature.com/articles/s41592-019-0494-8) | First to predict cell response to *unseen* perturbations via latent-space arithmetic. |
| Geneformer | Theodoris et al. | 2023 | [Nature](https://www.nature.com/articles/s41586-023-06139-9) | Rank-encoded transformer; in-silico perturbation of gene regulatory networks. |
| scGPT | Cui et al. | 2024 | [Nature Methods](https://www.nature.com/articles/s41592-024-02201-0) | Generative single-cell FM unifying multi-omics tasks; the reference baseline. |
| scFoundation | Hao et al. | 2024 | [Nature Methods](https://www.nature.com/articles/s41592-024-02305-7) | 100M-cell pretraining; strongest of the early masked-expression FMs on benchmarks. |
| Universal Cell Embedding (UCE) | Rosen et al. | 2024 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.28.568918) | Zero-shot, species-agnostic cell embeddings without per-dataset retraining. |
| Linear baselines critique | Ahlmann-Eltze et al. | 2025 | [Nature Methods](https://www.nature.com/articles/s41592-025-02772-6) | Deep perturbation predictors do not yet beat simple linear baselines — the field's reality check. |
| Tahoe-100M | Zhang et al. | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.20.639398) | 100M-cell drug-perturbation atlas (1,100 compounds × 50 lines); the fuel for perturbation FMs. |
| TranscriptFormer | Pearce et al. (CZI) | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.25.650731) | Generative cross-species atlas, 112M cells over 1.5B years of evolution; a queryable virtual cell. |
| Virtual Cell Challenge | Arc Institute | 2025 | [Cell](https://www.cell.com/cell/fulltext/S0092-8674(25)00675-0) | Defines a "Turing test" + live benchmark for perturbation prediction across contexts. |
| STATE | Arc Institute | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.26.661135) | First-gen virtual cell model; State Embedding + State Transition over 167M obs / 100M perturbed cells. |
| Tahoe-x1 | Tahoe / Arc | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.23.683759) | Scales perturbation-trained FMs to 3B parameters; current frontier on drug-response prediction. |

## Where it stands (2025-2026)
- **Perturbation prediction is the prize.** The frontier shifted from cell-type annotation/embedding to predicting the *transcriptional response* to genetic, chemical, and cytokine perturbations — and to unseen cell contexts. STATE and Tahoe-x1 lead.
- **Data became the bottleneck-breaker.** Arc's Virtual Cell Atlas (600M+ cells), Tahoe-100M, and AI-curated scBaseCount turned perturbation data from scarce to giga-scale, enabling supervised state-transition training.
- **Benchmarks got honest.** 2025 evaluations (Nature Methods, Genome Biology) showed masked-expression FMs often lose to HVG/scVI/logistic-regression and linear baselines zero-shot. The Virtual Cell Challenge now arbitrates with held-out screens.
- **Scaling kicked in.** Tahoe-x1 (3B params) shows perturbation-trained scaling laws are starting to bite — but only when trained on perturbation, not just observational, data.

## Transferred vs. reinvented

**Transferred from language modeling**
- Transformer backbone, masked-token pretraining, and large-scale self-supervision.
- Scaling-law mindset: more cells + parameters + perturbation diversity.
- Embeddings as a universal interface for downstream tasks and zero-shot transfer.

**Reinvented for cells**
- The "token": genes have no canonical order, so rank- or value-based expression encodings replace sequences.
- Perturbation as a first-class operator (State Transition), not a prompt.
- Cross-species / cross-context generalization as the core objective, with wet-lab screens — not next-token accuracy — as the loss that matters.

## Open problems
- **Do FMs beat linear baselines?** On many perturbation and annotation tasks, not yet robustly. Pretraining benefit is task- and protocol-dependent.
- **Generalizing to unseen perturbations and contexts** (new genes, new cell types, combinatorial perturbations) remains the hard, unsolved core.
- **Causality vs. correlation.** Models capture statistical structure; grounding predictions in mechanistic, causal biology is open.
- **Interpretability erodes** as models blackbox; linking latent state shifts to pathways/phenotype is largely missing.

## See also
- [06 · Protein & Biology](../06-protein-and-biology/README.md) — sequence-structure FMs and the wet-lab verification analogy.
- [07 · Omics & Genomics](../07-omics-and-genomics/README.md) — DNA/RNA sequence models that feed cellular state.
- [11 · Design Patterns](../11-design-patterns/README.md) — tokenization, masked pretraining, and scaling laws reused here.
- [Concept Library — Bricks](../../../library/bricks/README.md) — shared primitives across domains.
