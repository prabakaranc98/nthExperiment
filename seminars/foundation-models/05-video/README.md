# 05 · Video Foundation Models

*Temporal modeling. The world simulator hypothesis. Can a video model plan?*

**The key challenge:** video is image + time. The "token" is a spatiotemporal patch. The central research question: is a good video model implicitly a world model?

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| VideoGPT | Yan et al. | 2021 | [arXiv 2104.10157](https://arxiv.org/abs/2104.10157) | VQ-VAE + transformers for video generation |
| Video Generation as World Simulators (Sora) | OpenAI | 2024 | [Technical report](https://openai.com/research/video-generation-models-as-world-simulators) | Diffusion transformer over video; world simulator framing |
| Sora Review | Various | 2024 | [arXiv 2402.17177](https://arxiv.org/abs/2402.17177) | Background, technology, limitations |
| CogVideoX | THUDM | 2024 | [arXiv 2408.06072](https://arxiv.org/abs/2408.06072) | Expert transformer for text-to-video diffusion |
| V-JEPA | Bardes et al., Meta | 2024 | [arXiv 2404.08471](https://arxiv.org/abs/2404.08471) | Latent prediction in video; non-generative |
| V-JEPA 2 | Meta | 2025 | [arXiv 2506.09985](https://arxiv.org/abs/2506.09985) | Video models enable understanding, prediction, planning |
| GAIA-1 | Wayve | 2023 | [arXiv 2309.17080](https://arxiv.org/abs/2309.17080) | World model for autonomous driving from video |

**The critical question per paper:** Does this model generalize to novel situations, or does it interpolate? Is this a world model or a sophisticated pattern matcher?
**What video had to invent:** spatiotemporal patch embedding, temporal attention, causal masking over time, action conditioning.
