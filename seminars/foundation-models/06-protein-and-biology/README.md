# 06 · Protein & Biology Foundation Models

*The domain where prediction folds into design — and the molecule is the verifier.*

## The adaptation
- **Token** — an amino acid (sequence LM) or an atom / residue frame in 3D (structure model). Frontier models tokenize sequence, structure, and function jointly.
- **Objective** — masked-residue prediction (ESM line) or denoising/diffusion over coordinates (AlphaFold3, RFdiffusion). Evolution supplies the "corpus."
- **Inductive bias** — SE(3)-equivariance for 3D geometry; coevolution from multiple sequence alignments (MSAs); pairwise residue representations.
- **Verification signal** — does it fold, bind, express, catalyze? Wet-lab and physics-based ground truth make this the rare FM domain with a cheap, objective reward.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| AlphaFold2 | DeepMind | 2021 | [Nature 596](https://www.nature.com/articles/s41586-021-03819-2) | Cracked single-chain structure prediction at near-experimental accuracy. The landmark. |
| ProteinMPNN | Baker Lab | 2022 | [Science](https://www.science.org/doi/10.1126/science.add2187) | Inverse folding: structure → sequence. The workhorse of every design pipeline. |
| ESM2 / ESMFold | Meta | 2023 | [Science](https://www.science.org/doi/10.1126/science.ade2574) | Protein LM scaling laws; structure from a single sequence, no MSA. |
| RFdiffusion | Baker Lab | 2023 | [Nature](https://www.nature.com/articles/s41586-023-06415-8) | Diffusion over backbones; opened practical de novo design and motif scaffolding. |
| AlphaFold3 | DeepMind / Isomorphic | 2024 | [Nature](https://www.nature.com/articles/s41586-024-07487-w) | Diffusion over *all* biomolecules — proteins, ligands, nucleic acids, ions. |
| ESM3 | EvolutionaryScale | 2025 | [Science](https://www.science.org/doi/10.1126/science.ads0018) | Multimodal generative LM over sequence + structure + function; designed a novel fluorescent protein. |
| BindCraft | EPFL / MIT | 2025 | [Nature](https://www.nature.com/articles/s41586-025-09429-6) | One-shot binder design via AF2 backprop hallucination; 10–100% experimental hit rates. |
| Boltz-2 | MIT / Recursion | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1) | Open AF3-class structure *plus* binding affinity approaching FEP at ~1000× lower cost. |
| RFdiffusion2 | Baker Lab | 2025 | [Nature Methods](https://www.nature.com/articles/s41592-025-02975-x) | Atom-level enzyme active-site scaffolding from functional-group geometry alone. |
| salad | — | 2025 | [Nat. Mach. Intell.](https://www.nature.com/articles/s42256-025-01100-z) | Sparse all-atom denoising; up to 2 orders of magnitude faster generation than RFdiffusion. |
| RFdiffusion3 | Baker Lab | 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2.full) | All-atom de novo design of biomolecular interactions in one generative model. |

## Where it stands (2025-2026)
- **Beyond structure → affinity.** Boltz-2 fuses structure prediction with binding-affinity estimation, narrowing the gap to free-energy perturbation at a fraction of the compute.
- **Open AF3-class models are commoditized.** Boltz, Protenix, Chai, OpenFold3, and IntFold all match or approach AlphaFold3; the moat moved from architecture to data and affinity.
- **Design is one-shot and functional.** BindCraft and RFdiffusion2/3 push from "a backbone that folds" to "a binder/enzyme that works," often without high-throughput screening.
- **Dynamics is the new frontier.** Static-structure prediction is largely solved; conformational ensembles, allostery, and folding pathways are the open targets.

## Transferred vs. reinvented
**Transferred from language modeling**
- Masked-token pretraining and Transformer attention (ESM).
- Scaling laws: more sequences and parameters → emergent structural understanding.
- Multimodal prompting and generation (ESM3 over sequence/structure/function).

**Reinvented for biology**
- MSAs / coevolution as context, not free-text.
- SE(3)-equivariant attention and triangle updates for 3D geometry.
- Diffusion over atomic coordinates rather than discrete tokens.
- A real physical/experimental reward loop instead of human preference.

## Open problems
- **Conformational ensembles & dynamics** — single-structure outputs miss the states that govern function and drug binding.
- **Affinity calibration** — predicted binding still drifts from experiment; FEP-level accuracy is not yet general.
- **Design beyond well-behaved folds** — disordered regions, membrane proteins, and large assemblies remain hard.
- **Generalization vs. memorization** — how much is genuine physics versus recall of the PDB?

## See also
- [Omics & Genomics](../07-omics-and-genomics/README.md) — DNA/RNA language models, the sequence-to-function sibling.
- [Virtual Cells](../12-virtual-cells/README.md) — scaling from molecules to whole-cell state models.
- [Design Patterns](../11-design-patterns/README.md) — diffusion, equivariance, and verifiable-reward training shared across domains.
- [Concept Library](../../../library/bricks/README.md) — reusable bricks behind these models.
