# 12 · Virtual Cells

*Simulating cellular biology at scale. From single-cell FMs to whole-cell simulation.*

**The vision:** a "virtual cell" that can simulate the behavior of a cell under perturbations, predict drug responses, model disease states, and generate hypotheses for wet-lab experiments. The FM paradigm applied to cell biology: cells as systems to be modeled, not just sequences to be predicted.

**The "token" question:** a gene, a protein, a cell state, a perturbation? The answer depends on what you're modeling.

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| scGPT | Cui et al. | 2024 | [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02201-0) | Foundation model for single-cell multi-omics using generative AI |
| Geneformer | Theodoris et al. | 2023 | [Nature 2023](https://www.nature.com/articles/s41586-023-06139-9) | In silico perturbations; predicts gene regulatory network effects |
| scFoundation | Hao et al. | 2024 | [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02305-7) | 100M-cell pretraining for single-cell analysis |
| CellLM (Cell Language Model) | Zhao et al. | 2023 | [arXiv 2306.04371](https://arxiv.org/abs/2306.04371) | Unified cell representation learning |
| A Virtual Cell — Recursion / Allen Institute vision | Allen Institute | 2024 | [Cell 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01221-X) | Manifesto for a whole-cell simulation FM |
| AIDO.Cell | ARC Institute | 2024 | [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.12.23.630009) | Single-cell foundation model across 50M cells |
| Perturbation modeling with scGen | Lotfollahi et al. | 2019 | [Nature Methods 2019](https://www.nature.com/articles/s41592-019-0494-8) | Predicting cell response to unseen perturbations |

**The grand challenge:** move from predicting gene expression to predicting *phenotype* — what happens to a cell under a drug, a genetic perturbation, a disease state. This requires grounding the FM in causal biology, not just statistical patterns.

**The verification signal:** wet-lab experiments (CRISPR screens, drug response assays) provide ground truth — similar to protein folding's advantage.
