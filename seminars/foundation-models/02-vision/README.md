# 02 · Vision Foundation Models

*Pixels become patches, patches become tokens — and self-supervision turned labels optional.*

## The adaptation
- **Token:** a non-overlapping image patch (typically 14×14 or 16×16 px), linearly projected to an embedding.
- **Pretraining objective:** masked-patch reconstruction (MAE), self-distillation (DINO), or image–text contrast (CLIP/SigLIP) — supervised ImageNet labels are no longer the driver.
- **Inductive bias:** weak by design. ViT discards convolutional locality and learns 2D structure from data via positional embeddings; scale substitutes for built-in priors.
- **Verification signal:** frozen-feature transfer — linear probing, dense prediction (segmentation, depth), and zero-shot retrieval measure whether representations generalize without fine-tuning.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| ViT — *An Image is Worth 16×16 Words* | Google | 2020 | [arXiv 2010.11929](https://arxiv.org/abs/2010.11929) | Patches as tokens; at scale, a plain transformer beats convolutional priors |
| CLIP | OpenAI | 2021 | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) | Web-scale image–text contrast → zero-shot transfer |
| MAE | Meta | 2021 | [arXiv 2111.06377](https://arxiv.org/abs/2111.06377) | 75% masking makes generative SSL scale on images |
| DINOv2 | Meta | 2023 | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) | Self-supervised features rival supervised; emergent segmentation |
| SAM | Meta | 2023 | [arXiv 2304.02643](https://arxiv.org/abs/2304.02643) | Promptable universal segmentation; the first vision "foundation" interface |
| SAM 2 | Meta | 2024 | [arXiv 2408.00714](https://arxiv.org/abs/2408.00714) | Streaming memory extends promptable segmentation to video |
| AIMv2 | Apple | 2024 | [arXiv 2411.14402](https://arxiv.org/abs/2411.14402) | Autoregressive multimodal pretraining beats contrastive encoders for VLM use |
| SigLIP 2 | Google DeepMind | 2025 | [arXiv 2502.14786](https://arxiv.org/abs/2502.14786) | Sigmoid loss + captioning + self-distillation; default encoder for Gemma 3, Qwen3-VL |
| V-JEPA 2 | Meta | 2025 | [arXiv 2506.09985](https://arxiv.org/abs/2506.09985) | Latent-space prediction over video → world model for planning |
| DINOv3 | Meta | 2025 | [arXiv 2508.10104](https://arxiv.org/abs/2508.10104) | 7B ViT on 1.7B images; Gram anchoring fixes dense-feature decay at scale |
| SAM 3 | Meta | 2025 | [arXiv 2511.16719](https://arxiv.org/abs/2511.16719) | Promptable *concept* segmentation — noun phrases and exemplars, not just clicks |

## Where it stands (2025-2026)
- **DINOv3** is the SSL frontier: a 7B ViT trained label-free on 1.7B images, with **Gram anchoring** curing the long-known degradation of dense feature maps over long schedules. It beats prior self- and weakly-supervised models on ADE20k segmentation (+6 mIoU) and depth, and distills cleanly into ViT-S/B/L and ConvNeXt.
- **SAM 3** (Nov 2025) shifts segmentation from geometric prompts to **open-vocabulary concepts** — short noun phrases or image exemplars — roughly doubling accuracy on the new SA-CO benchmark; **SAM 3.1** (Mar 2026) adds faster joint multi-object tracking.
- **SigLIP 2** has become the de facto vision encoder feeding open VLMs (Gemma 3, Qwen3-VL), with 109-language coverage and stronger localization/dense features.
- **JEPA-style latent prediction** (V-JEPA 2) is the live bet against pixel/contrastive objectives: predict in representation space, target robotic planning and physical-world understanding rather than classification accuracy.

## Transferred vs. reinvented
**Transferred directly from language modeling:**
- The transformer block and attention, essentially unchanged.
- Scaling laws — more data and parameters keep paying off.
- Masked-prediction self-supervision (MAE ≈ a visual MLM).
- The "pretrain once, prompt/probe everywhere" foundation-model recipe.

**Invented fresh for vision:**
- Patch tokenization and 2D/3D positional encodings (incl. spatiotemporal RoPE for video).
- Heavy view augmentation as the core signal for contrastive and self-distillation methods.
- The masking-ratio puzzle — images need ~75% masking, far above text.
- Promptable segmentation interfaces and the data engines (SA-1B, SA-CO) that bootstrap them.
- Latent-space (JEPA) objectives that avoid reconstructing pixels at all.

## Open problems
- **Dense vs. global features:** strong image-level embeddings still don't guarantee clean per-pixel features; Gram anchoring helps but the trade-off persists.
- **Spatial reasoning:** VLM encoders remain weak at counting, relative position, and fine localization despite strong retrieval scores.
- **Objective uncertainty:** no consensus among contrastive, masked-generative, and latent-predictive (JEPA) pretraining — each wins different downstream tasks.
- **Evaluation drift:** ImageNet linear-probe is saturated and a poor proxy for real transfer to dense, video, and embodied tasks.

## See also
- [Language Foundation Models](../01-language/README.md) — the transformer and scaling recipe vision borrowed.
- [Video Foundation Models](../05-video/README.md) — SAM 2 and V-JEPA 2 push patches into the temporal axis.
- [Multimodal Foundation Models](../10-multimodal/README.md) — where SigLIP 2 and AIMv2 encoders plug into VLMs.
- [Concept Library](../../../library/bricks/README.md) — patch tokenization, contrastive learning, masked prediction.
