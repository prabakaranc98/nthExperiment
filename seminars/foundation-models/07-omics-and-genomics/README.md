# 07 · Omics & Genomics Foundation Models

*DNA/RNA as language. Single-cell transcriptomics. The multi-omics challenge.*

**The key adaptation:** nucleotide sequences (A, T, C, G) are discrete tokens — closer to language than protein. But the "meaning" of a sequence depends on position, context (regulatory elements), and organism. Single-cell RNA-seq adds another dimension: gene expression as a distribution.

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Nucleotide Transformer | Dalla et al., InstaDeep | 2024 | [Nature Methods](https://www.nature.com/articles/s41592-024-02523-z) | Human genomics foundation model; multi-species |
| DNABERT-2 | Zhou et al. | 2023 | [arXiv 2306.15006](https://arxiv.org/abs/2306.15006) | Multi-species DNA FM with efficient tokenization |
| HyenaDNA | Poli et al. | 2023 | [arXiv 2306.15794](https://arxiv.org/abs/2306.15794) | Sub-quadratic SSM for single-nucleotide resolution |
| Effective gene expression prediction from sequence (Enformer) | Avsec et al., DeepMind | 2021 | [Nature Methods 2021](https://www.nature.com/articles/s41592-021-01252-x) | Long-range regulatory interactions |
| Geneformer | Theodoris et al. | 2023 | [Nature 2023](https://www.nature.com/articles/s41586-023-06139-9) | Single-cell FM; rank-value encoding of gene expression |
| scGPT | Cui et al. | 2024 | [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02201-0) | Foundation model for single-cell multi-omics |
| Evo | Arc Institute | 2024 | [arXiv 2410.18966](https://arxiv.org/abs/2410.18966) | DNA FM at single-nucleotide resolution, 1M context |

**What genomics had to invent:** tokenization of long DNA sequences (k-mer vs. BPE vs. character), modeling regulatory grammar at long range, handling cell-type specificity.
**The unique challenge:** context matters enormously — the same sequence means different things in different genomic contexts.
