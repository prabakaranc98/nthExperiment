# 08 · Physical Sciences Foundation Models

*One model for the equations of motion, one model for the periodic table.*

## The adaptation
- **Token** — a grid/mesh node and its physical fields (PDEs), or an atom with its position and species (atomistics). Geometry, not sequence order, defines neighbors.
- **Objective** — learn the *solution operator* (map initial/boundary conditions to future fields) or the *potential energy surface* (map atomic geometry to energy and forces). Both are next-state regression, not next-token classification.
- **Inductive bias** — physical symmetry is non-negotiable: E(3) equivariance for atoms, resolution/discretization invariance for fields, conservation laws baked into the architecture or loss.
- **Verification signal** — the ground truth is computable. DFT relaxations, MD stability, and PDE residuals provide cheap, abundant labels and let active learning close the loop.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| DeepONet | Brown | 2019 | [arXiv 1910.03193](https://arxiv.org/abs/1910.03193) | Universal approximation for operators; branch/trunk split |
| Fourier Neural Operator (FNO) | Caltech | 2020 | [arXiv 2010.08895](https://arxiv.org/abs/2010.08895) | Learns PDE operators in Fourier space; resolution-invariant |
| MACE | Cambridge | 2022 | [arXiv 2206.07697](https://arxiv.org/abs/2206.07697) | Higher-order equivariant message passing; the default force-field backbone |
| GNoME | Google DeepMind | 2023 | [Nature](https://www.nature.com/articles/s41586-023-06735-9) | 2.2M stable crystals via GNN + DFT active learning |
| MatterSim | Microsoft | 2024 | [arXiv 2405.04967](https://arxiv.org/abs/2405.04967) | Universal potential across the periodic table, 0–5000 K, ≤1000 GPa |
| Poseidon | ETH Zürich | 2024 | [arXiv 2405.19101](https://arxiv.org/abs/2405.19101) | Multiphysics-pretrained PDE foundation model; strong few-shot transfer |
| OMat24 + eSEN | Meta FAIR | 2024 | [arXiv 2410.12771](https://arxiv.org/abs/2410.12771) | 110M-DFT dataset; tops Matbench-Discovery (F1 > 0.9) |
| MatterGen | Microsoft | 2025 | [Nature](https://www.nature.com/articles/s41586-025-08628-5) | Diffusion over crystals; property-guided inverse design ([arXiv 2312.03687](https://arxiv.org/abs/2312.03687)) |
| Orb-v3 | Orbital Materials | 2025 | [arXiv 2504.06231](https://arxiv.org/abs/2504.06231) | Non-equivariant, non-conservative potential; ~10× faster at near-SOTA accuracy |
| UMA + OMol25 | Meta FAIR | 2025 | [arXiv 2506.23971](https://arxiv.org/abs/2506.23971) | One Mixture-of-Linear-Experts model across molecules, catalysts, materials ([OMol25](https://arxiv.org/abs/2505.08762)) |
| Walrus | Polymathic AI | 2025 | [arXiv 2511.15684](https://arxiv.org/abs/2511.15684) | 1.3B-param continuum-dynamics model pretrained on 19 physical scenarios |

## Where it stands (2025-2026)
- **One model for all of chemistry.** Meta's **UMA** unifies molecules, materials, and catalysts in a single interatomic potential; **OMol25** (100M+ DFT records) is the new high-accuracy pretraining substrate.
- **The equivariance tax is being questioned.** **Orb-v3** shows non-equivariant, non-conservative architectures can match accuracy with ~10× lower latency — scale and data may substitute for hard-coded symmetry.
- **PDE foundation models scaled up.** **Walrus** (1.3B params) and follow-ups push cross-domain continuum dynamics; benchmarks like **The Well** make multiphysics transfer measurable.
- **Inverse design is operational.** **MatterGen**'s Nature publication and RL/guidance-fine-tuned variants generate stable crystals to spec (band gap, magnetism, mechanics), not just screen existing ones.

## Transferred vs. reinvented
**Transferred directly from language modeling**
- Transformer backbones, scaling laws, and large-scale self-supervised pretraining then fine-tuning.
- Diffusion/flow-matching generative recipes, adapted to lattices and atom clouds.
- Mixture-of-experts for capacity without proportional compute (UMA).

**Invented fresh here**
- E(3)/SE(3)-equivariant layers and spherical-harmonic message passing.
- Resolution- and discretization-invariant operators (spectral, Fourier).
- Energy-conserving force prediction via gradients of a learned potential.
- DFT/MD-in-the-loop active learning to generate labels at scale.

## Open problems
- **Out-of-distribution stability.** Potentials drift or break in long MD rollouts and on chemistries far from the training manifold.
- **Equivariance vs. scale.** Open question whether learned symmetry from data robustly replaces architectural symmetry, especially for high-order derivatives.
- **Synthesizability gap.** Generative models propose "stable" crystals faster than labs can make them; thermodynamic stability ≠ a viable synthesis route.
- **Trustworthy uncertainty.** Calibrated error bars for surrogate predictions remain weak, limiting use in high-stakes design and autonomous labs.

## See also
- [../03-time-series/README.md](../03-time-series/README.md) — sequential dynamics and forecasting operators
- [../06-protein-and-biology/README.md](../06-protein-and-biology/README.md) — equivariant structure prediction for biomolecules
- [../13-climate-and-earth/README.md](../13-climate-and-earth/README.md) — large-scale PDE emulation for weather and climate
- [../../../library/bricks/README.md](../../../library/bricks/README.md) — shared concepts (equivariance, diffusion, scaling laws)
