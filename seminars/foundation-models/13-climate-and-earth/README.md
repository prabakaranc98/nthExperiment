# 13 · Climate & Earth Foundation Models

*The verification-rich domain: tomorrow is the test set, and AI now beats the supercomputer.*

## The adaptation
- **The "token":** a grid cell on the sphere at a given pressure level — ~10⁵–10⁶ cells per atmospheric state, ~80 variables.
- **Pretraining objective:** autoregressive next-state prediction (6h step) on ERA5 reanalysis; the frontier has shifted from deterministic regression to **diffusion sampling** of the forecast distribution.
- **Key inductive bias:** geometry-aware operators — spherical harmonics (SFNO), icosahedral GNNs (GraphCast), or 3D Swin/ViT on the lat-lon grid. Physics is encoded as architecture, not loss terms.
- **Verification signal:** the atmosphere itself. Skill is scored against held-out reanalysis and operational analyses on standard variables (Z500, T850, 10m wind) via [WeatherBench 2](https://sites.research.google/weatherbench/).

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| FourCastNet | NVIDIA | 2022 | [arXiv 2202.11214](https://arxiv.org/abs/2202.11214) | First global AFNO forecaster; proved data-driven weather at 0.25° is viable |
| Pangu-Weather | Huawei | 2023 | [Nature](https://www.nature.com/articles/s41586-023-06185-3) | First ML model to beat ECMWF HRES on medium-range deterministic skill |
| GraphCast | Google DeepMind | 2023 | [Science](https://www.science.org/doi/10.1126/science.adi2336) | Icosahedral GNN; outperforms HRES on >90% of 10-day targets — the canonical baseline |
| ClimaX | UCLA / Microsoft | 2023 | [arXiv 2301.10343](https://arxiv.org/abs/2301.10343) | ViT backbone pretrained for both weather *and* climate tasks; early "FM" framing |
| NeuralGCM | Google | 2024 | [Nature](https://www.nature.com/articles/s41586-024-07744-y) | Differentiable dynamical core + learned subgrid physics; stable decadal climate runs |
| GenCast | Google DeepMind | 2024 | [Nature](https://www.nature.com/articles/s41586-024-09032-1) · [arXiv 2312.15796](https://arxiv.org/abs/2312.15796) | Diffusion ensemble beats ECMWF ENS on 97% of probabilistic targets — set the new bar |
| Prithvi-WxC | NASA / IBM | 2024 | [arXiv 2409.13598](https://arxiv.org/abs/2409.13598) | 2.3B atmospheric FM with masked + forecast pretraining; downscaling/gravity-wave finetunes |
| ACE2 (Climate Emulator) | Ai2 | 2024 | [arXiv 2411.11268](https://arxiv.org/abs/2411.11268) | SFNO emulator stable over 100s of years; reproduces forced climate response |
| Aurora | Microsoft | 2025 | [Nature](https://www.nature.com/articles/s41586-025-09005-y) · [arXiv 2405.13063](https://arxiv.org/abs/2405.13063) | 1.3B FM finetuned to air quality, ocean waves, cyclones — true cross-task transfer |
| AIFS ENS | ECMWF | 2025 | [arXiv 2509.18994](https://arxiv.org/abs/2509.18994) | Official 51-member ML ensemble, **operational since July 2025** alongside the IFS |
| cBottle ("Climate in a Bottle") | NVIDIA / MPI-M | 2025 | [arXiv 2505.06474](https://arxiv.org/abs/2505.06474) | Generative FM sampling km-scale global climate states; ~3000× data compression |

## Where it stands (2025-2026)
- **AI is operational, not experimental.** ECMWF runs **AIFS ENS** in production; NOAA deployed multiple AI global models in late 2025. The question is no longer "can it beat physics" but "which AI model."
- **Diffusion ensembles won the medium range.** GenCast-style probabilistic models dominate extreme-event and tail-risk skill; deterministic regression models blur at long lead times.
- **One FM, many tasks.** Aurora demonstrates that a single pretrained earth-system model finetunes to air quality, ocean waves, and tropical cyclones — the foundation-model thesis realized for geophysics.
- **The frontier is generative climate.** cBottle and ACE2 move from *forecasting* (initial-value problem) to *emulating the climate distribution* at km-scale — sampling plausible states rather than rolling one out.
- **Hybrid is resurgent.** NeuralGCM and ECMWF's "ML-augmented IFS" roadmap keep a differentiable physics core, trading peak data-driven skill for long-run stability and conservation.

## Transferred vs. reinvented
**Transferred from language/vision:**
- Autoregressive next-state prediction as the pretraining objective.
- Transformer / attention backbones (3D Swin in Aurora, ViT in ClimaX) and masked-token pretraining (Prithvi-WxC).
- Diffusion sampling, borrowed wholesale from image generation, for ensembles.

**Invented fresh here:**
- Spherical-geometry operators (SFNO, icosahedral GNNs) — the data lives on a sphere, not a flat grid.
- ERA5 reanalysis as a 40-year, physically-consistent "pretraining corpus" assimilated from observations.
- Stability constraints for unbounded autoregressive rollouts (climate runs span centuries, not 10 days).
- Verification against operational analyses and physics-derived metrics (energy spectra, conservation).

## Open problems
- **Extrapolation vs. interpolation:** trained on the historical distribution, can these models forecast unseen warming regimes — or only interpolate within it?
- **Long-rollout drift:** autoregressive models blur toward climatology and can violate conservation laws over long horizons.
- **Spectral blurring:** deterministic models systematically under-represent fine-scale variance; ensembles help but cost compute.
- **Coupling & extremes:** full atmosphere-ocean-land coupling and reliable rare-event statistics (storm intensity, precipitation tails) remain hard.

## See also
- [`../03-time-series/README.md`](../03-time-series/README.md) — forecasting backbones and probabilistic sequence models
- [`../08-physical-sciences/README.md`](../08-physical-sciences/README.md) — neural operators and differentiable simulators (SFNO, PDE solvers)
- [`../02-vision/README.md`](../02-vision/README.md) — ViT/Swin backbones reused for gridded earth-observation data
- [`../../../library/bricks/README.md`](../../../library/bricks/README.md) — diffusion, masked pretraining, and scaling-law concepts
