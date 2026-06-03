# 04 · Audio & Speech Foundation Models

*Self-supervised speech representations. Generative audio at scale. The path from recognition to generation.*

**The key adaptation:** raw audio → discrete tokens (via codec) or continuous representations (via SSL). Speech has rich temporal structure at multiple timescales (phoneme, word, prosody, speaker).

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| wav2vec 2.0 | Baevski et al., Meta | 2020 | [arXiv 2006.11477](https://arxiv.org/abs/2006.11477) | Contrastive masked SSL for speech; the BERT of audio |
| HuBERT | Hsu et al., Meta | 2021 | [arXiv 2106.07447](https://arxiv.org/abs/2106.07447) | Offline clustering → masked prediction |
| Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford et al., OpenAI | 2022 | [arXiv 2212.04356](https://arxiv.org/abs/2212.04356) | Supervised at scale; multilingual ASR |
| High Fidelity Neural Audio Compression (EnCodec) | Défossez et al., Meta | 2022 | [arXiv 2210.13438](https://arxiv.org/abs/2210.13438) | Neural audio codec; enables discrete audio tokens |
| AudioLM | Zeghidour et al., Google | 2022 | [arXiv 2209.03143](https://arxiv.org/abs/2209.03143) | Language modeling over audio tokens |
| Simple and Controllable Music Generation (MusicGen) | Copet et al., Meta | 2023 | [arXiv 2306.05284](https://arxiv.org/abs/2306.05284) | Text-conditioned music generation |
| Voicebox | Le et al., Meta | 2023 | [arXiv 2306.15687](https://arxiv.org/abs/2306.15687) | Flow matching for speech; multilingual |
| AudioCraft | Meta FAIR | 2023 | [GitHub](https://github.com/facebookresearch/audiocraft) | Unified framework: MusicGen + AudioGen |
| WavLM | Chen et al., Microsoft | 2021 | [arXiv 2110.13900](https://arxiv.org/abs/2110.13900) | Masked speech + denoising; SUPERB SOTA |

**What audio had to invent:** discrete audio tokenization (EnCodec), multi-scale temporal modeling, disentangling speaker / content / prosody.
**What transferred:** transformer backbone, masked prediction objective, contrastive learning.
