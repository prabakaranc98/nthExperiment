# 02 · Vision Foundation Models

*Patches as tokens. What SAM proved about universal segmentation. The ViT scaling story.*

**The key adaptation:** images → non-overlapping patches → linear projection → transformer. The "token" is a patch. The pretraining objective shifted from supervised (ImageNet) to self-supervised (MAE, DINO) and contrastive (CLIP).

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| An Image is Worth 16×16 Words (ViT) | Dosovitskiy et al. | 2020 | [arXiv 2010.11929](https://arxiv.org/abs/2010.11929) | Patches as tokens; scale beats inductive bias |
| Learning Transferable Visual Models (CLIP) | Radford et al. | 2021 | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) | Image-text contrastive at web scale |
| Masked Autoencoders (MAE) | He et al. | 2021 | [arXiv 2111.06377](https://arxiv.org/abs/2111.06377) | 75% masking ratio; surprisingly effective |
| DINO | Caron et al. | 2021 | [arXiv 2104.14294](https://arxiv.org/abs/2104.14294) | Self-distillation → emergent segmentation |
| DINOv2 | Oquab et al. | 2023 | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) | Self-supervised features matching supervised at scale |
| Segment Anything (SAM) | Kirillov et al. | 2023 | [arXiv 2304.02643](https://arxiv.org/abs/2304.02643) | Universal segmentation; promptable vision |
| SAM 2 | Ravi et al. | 2024 | [arXiv 2408.00714](https://arxiv.org/abs/2408.00714) | SAM extended to video |
| SigLIP | Zhai et al. | 2023 | [arXiv 2303.15343](https://arxiv.org/abs/2303.15343) | Sigmoid loss removes global normalization |
| Florence-2 | Xiao et al. | 2023 | [arXiv 2311.06242](https://arxiv.org/abs/2311.06242) | Unified vision model across tasks |
| EVA | Fang et al. | 2023 | [arXiv 2211.07636](https://arxiv.org/abs/2211.07636) | Scaling masked visual pretraining |

**What vision had to invent:** positional encodings for 2D grids, patch tokenization, augmentation strategies for contrastive learning, the masking ratio puzzle.
**What transferred directly:** transformer architecture, scaling laws, self-supervised objectives (masked prediction).
