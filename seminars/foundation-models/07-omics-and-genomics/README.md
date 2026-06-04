# 07 · Omics & Genomics Foundation Models

*Two languages in one cell: the genome's nucleotides and the transcriptome's expression states.*

## The adaptation
- **Token.** Two regimes. **DNA/RNA:** nucleotides (single-base) or learned BPE k-mers — a 4-letter alphabet with regulatory grammar spread over megabases. **Single-cell:** genes are tokens; expression is encoded as rank-order (Geneformer) or binned/continuous values (scGPT).
- **Objective.** Causal or masked next-token prediction on raw sequence (DNA models); masked gene/value imputation over a cell's expression vector (single-cell models); perturbation-response prediction (virtual cells).
- **Inductive bias.** Long-range dependency at single-base resolution → SSM/convolutional hybrids (Hyena/StripedHyena) escape attention's quadratic cost. Single-cell models drop positional order: a cell is a *set* of genes, not a sequence.
- **Verification signal.** Zero-shot variant-effect prediction (clinical BRCA1, ClinVar, QTLs), cell-type annotation, perturbation response — measured against wet-lab ground truth, not held-out likelihood.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| Enformer | DeepMind | 2021 | [Nature Methods](https://www.nature.com/articles/s41592-021-01252-x) | Cracked long-range (100kb) sequence→expression; the regulatory-genomics baseline |
| Geneformer | Theodoris et al. | 2023 | [Nature](https://www.nature.com/articles/s41586-023-06139-9) | First single-cell FM; rank-value encoding, zero-shot network biology |
| HyenaDNA | Poli et al. | 2023 | [arXiv 2306.15794](https://arxiv.org/abs/2306.15794) | Sub-quadratic SSM; single-nucleotide resolution to 1M context |
| scFoundation | Hao et al. | 2023 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.05.29.542705) | 100M-param, read-depth-aware pretraining on 50M cells |
| Nucleotide Transformer | InstaDeep | 2024 | [Nature Methods](https://www.nature.com/articles/s41592-024-02523-z) | Multi-species DNA FM; strong transfer across genomic tasks |
| scGPT | Cui et al. | 2024 | [Nature Methods](https://www.nature.com/articles/s41592-024-02201-0) | 33M-cell generative FM; annotation, batch integration, perturbation |
| Evo 2 | Arc Institute / NVIDIA | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1) | 40B params, 9.3T bp, 1M context, all domains of life; zero-shot variant effects |
| TranscriptFormer | CZI | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1) | Generative cross-species atlas; 112M cells, 12 species, 1.5B years of evolution |
| AlphaGenome | DeepMind | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1) | 1Mb input → 1000s of base-resolution tracks; SOTA on 24/26 VEP benchmarks |
| STATE | Arc Institute | 2025 | [Arc Institute](https://arcinstitute.org/news/virtual-cell-model-state) | First virtual-cell model; predicts perturbation response from 170M+ cells |

## Where it stands (2025-2026)
- **DNA scaling worked.** Evo 2 (40B, 9.3T bp) generalizes across the tree of life and predicts variant pathogenicity with no task-specific training; published in [Nature 2026](https://www.nature.com/articles/s41586-026-10176-5).
- **Regulatory genomics has a new SOTA.** [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) unifies expression, splicing, chromatin, and TF binding in one 1Mb model — the strongest non-coding variant interpreter to date.
- **The frontier shifted to the *virtual cell*.** Arc's STATE and the [Virtual Cell Challenge](https://www.cell.com/cell/fulltext/S0092-8674(25)00675-0) reframe the goal as predicting perturbation response; the Tahoe–Arc–Biohub partnership (Jan 2026) is generating the largest perturbation dataset yet.
- **Cross-species is the new axis.** TranscriptFormer transfers cell-type biology across 685M+ years of divergence — generalization, not just bigger human atlases.

## Transferred vs. reinvented
**Transferred directly from language modeling**
- Causal/masked pretraining, the transformer stack, and zero-shot transfer to downstream tasks.
- BPE-style tokenization (DNABERT-2, NT) and scaling-law intuitions.

**Reinvented for omics**
- Sub-quadratic SSM/convolutional hybrids (Hyena, StripedHyena 2) for megabase context at single-base resolution.
- Order-free, set-based encodings of a cell (rank-value, expression binning) — there is no sentence to read.
- Perturbation-conditioned generation: predict the *counterfactual* cell state after a genetic or chemical intervention.

## Open problems
- **Perturbation prediction still trails naive baselines** on many Virtual Cell Challenge metrics — generalizing to *unseen* perturbations is unsolved.
- **Genomic LMs underperform on some clinical tasks** versus specialized supervised models; the FM advantage is uneven across benchmarks.
- **Batch effects and atlas bias:** single-cell FMs can memorize technical artifacts rather than biology.
- **Interpretability of regulatory grammar:** which learned features are causal vs. correlational remains hard to verify.

## See also
- [../06-protein-and-biology/README.md](../06-protein-and-biology/README.md) — sequence-to-structure; the sibling biological FM
- [../12-virtual-cells/README.md](../12-virtual-cells/README.md) — perturbation prediction and cell-state simulation
- [../10-multimodal/README.md](../10-multimodal/README.md) — fusing sequence, expression, and imaging modalities
- [../../../library/bricks/README.md](../../../library/bricks/README.md) — shared concepts (tokenization, scaling laws, SSMs)
