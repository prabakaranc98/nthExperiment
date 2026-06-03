# 06 · Protein & Biology Foundation Models

*From structure prediction to generative design. The most dramatic success of FM outside language.*

**The key insight:** amino acid sequences are like sentences; protein folding is prediction; protein design is generation. The "token" is an amino acid or 3D structure element. The verifiable reward (does it fold/bind/express?) makes this the ideal FM domain.

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| AlphaFold2 | Jumper et al., DeepMind | 2021 | [Nature 596](https://www.nature.com/articles/s41586-021-03819-2) | Solved protein structure prediction; landmark |
| ESM2 | Lin et al., Meta | 2023 | [Science 2023](https://www.science.org/doi/10.1126/science.ade2574) | Protein LM → atomic structure at scale |
| ESM3 | EvolutionaryScale | 2024 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.01.600583) | Multimodal: sequence + structure + function |
| Robust deep learning-based protein sequence design (ProteinMPNN) | Dauparas et al. | 2022 | [Science 2022](https://www.science.org/doi/10.1126/science.add2187) | Inverse folding: structure → sequence |
| De novo design with RFDiffusion | Watson et al. | 2023 | [Nature 2023](https://www.nature.com/articles/s41586-023-06415-8) | Diffusion for protein backbone design |
| AlphaFold3 | Abramson et al., DeepMind | 2024 | [Nature 2024](https://www.nature.com/articles/s41586-024-07487-w) | Diffusion over all biomolecular structures |
| Boltz-1 | MIT / EvolutionaryScale | 2024 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.19.624167) | Open AF3-level biomolecular structure prediction |
| Chai-1 | Chai Discovery | 2024 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.10.615955) | Molecular complex structure prediction |

**What biology had to invent:** multiple sequence alignment as context, structure-aware attention, equivariant networks for 3D geometry.
**The verification signal advantage:** biological ground truth (does it fold? does it bind?) makes RL and active learning much more tractable than in language.
