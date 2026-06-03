# 10 · Multimodal Foundation Models

*Cross-modal alignment. Any-to-any. Binding modalities into shared representation space.*

**The key question:** how do you build a model that understands and generates across modalities? Three paradigms: (1) contrastive alignment (CLIP), (2) frozen LLM + adapter (Flamingo, LLaVA), (3) unified generative (any-to-any tokenization).

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Flamingo | Alayrac et al., DeepMind | 2022 | [arXiv 2204.14198](https://arxiv.org/abs/2204.14198) | Frozen LLM + visual adapter; few-shot VLM |
| Visual Instruction Tuning (LLaVA) | Liu et al. | 2023 | [arXiv 2304.08485](https://arxiv.org/abs/2304.08485) | GPT-4 generated instruction tuning for VLMs |
| Unified-IO 2 | Lu et al. | 2023 | [arXiv 2312.17172](https://arxiv.org/abs/2312.17172) | Vision + language + audio + action in one model |
| 4M: Massively Multimodal Masked Modeling | Mizrahi et al., EPFL | 2023 | [arXiv 2312.06647](https://arxiv.org/abs/2312.06647) | Any-to-any generation across modalities |
| 4M-21 | EPFL | 2024 | [arXiv 2406.09406](https://arxiv.org/abs/2406.09406) | Scaling 4M to 21 modalities |
| ImageBind | Girdhar et al., Meta | 2023 | [arXiv 2305.05665](https://arxiv.org/abs/2305.05665) | One embedding space binding image, text, audio, depth, IMU, thermal |
| Meta-Transformer | Gong et al. | 2023 | [arXiv 2307.10802](https://arxiv.org/abs/2307.10802) | Unified framework for 12 modalities |
| Chameleon (early fusion) | Meta | 2024 | [arXiv 2405.09818](https://arxiv.org/abs/2405.09818) | Mixed-modal early fusion; token-level interleaving |

**The design space question:** late fusion (independent encoders → LLM) vs. early fusion (all tokens together from the start)? Late fusion is cheaper; early fusion learns richer cross-modal interactions.
