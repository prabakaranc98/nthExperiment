# 08 · Physical Sciences — Neural Operators & Materials

*Learning solution operators for PDEs. Materials discovery at scale. Physics-constrained generation.*

**The key adaptation:** physical systems have symmetries (translation, rotation, equivariance) that must be respected. The "token" is a grid point, atom, or mesh node. The training objective is to learn the *operator* that maps inputs (initial conditions, parameters) to outputs (solutions).

## Paper Log

### Neural Operators (PDEs)

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Fourier Neural Operator (FNO) | Li et al., Caltech | 2020 | [arXiv 2010.08895](https://arxiv.org/abs/2010.08895) | Learn PDE solution operators in Fourier space |
| DeepONet | Lu et al. | 2019 | [arXiv 1910.03193](https://arxiv.org/abs/1910.03193) | Universal approximation for operators |
| Poseidon | Subramanian et al. | 2024 | [arXiv 2405.19101](https://arxiv.org/abs/2405.19101) | PDE foundation model; multiphysics pretraining |
| CoDA-NO | Rahman et al. | 2024 | [arXiv 2403.12553](https://arxiv.org/abs/2403.12553) | Codomain attention for multiphysics |
| Physics-Informed Neural Networks (PINNs) | Raissi et al. | 2019 | [Journal paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125) | Enforce physical laws as soft constraints |

### Materials Science

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Scaling deep learning for materials discovery (GNoME) | Merchant et al., DeepMind | 2023 | [Nature 2023](https://www.nature.com/articles/s41586-023-06735-9) | 2.2M stable materials; GNN + DFT active learning |
| MatterGen | Zeni et al., Microsoft | 2023 | [arXiv 2312.03687](https://arxiv.org/abs/2312.03687) | Diffusion over crystal structures; inverse design |
| E(3)-equivariant GNNs for interatomic potentials (NequIP) | Batzner et al. | 2021 | [arXiv 2101.03164](https://arxiv.org/abs/2101.03164) | Equivariant atomic potential learning |
| MACE | Batatia et al. | 2022 | [arXiv 2206.07697](https://arxiv.org/abs/2206.07697) | Higher-order equivariant message passing for force fields |

**What physical sciences had to invent:** equivariant networks (respecting physical symmetries), resolution invariance, multiscale architectures, operator-level generalization.
