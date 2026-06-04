# 04 · Audio & Speech Foundation Models

*From recognizing speech to speaking back — audio became a token stream the transformer could read and write.*

## The adaptation
- **Token:** either continuous SSL features (wav2vec/HuBERT) for understanding, or discrete codec tokens (EnCodec/Mimi) for generation — modern systems stack both: semantic tokens for content, acoustic tokens for fidelity.
- **Objective:** masked prediction of cluster/quantizer targets (SSL) for representations; next-token or flow-matching/diffusion over codec latents for synthesis.
- **Inductive bias:** multi-timescale structure — phoneme → word → prosody → speaker — and the need to disentangle *what* is said from *who* says it and *how*.
- **Verification signal:** WER for ASR; speaker/intelligibility and MOS-style perceptual scores for TTS; full-duplex latency and turn-taking for live dialogue.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| wav2vec 2.0 | Meta | 2020 | [arXiv 2006.11477](https://arxiv.org/abs/2006.11477) | Contrastive masked SSL; the BERT moment for speech representations |
| HuBERT | Meta | 2021 | [arXiv 2106.07447](https://arxiv.org/abs/2106.07447) | Offline clustering → masked prediction; the dominant SSL recipe |
| WavLM | Microsoft | 2021 | [arXiv 2110.13900](https://arxiv.org/abs/2110.13900) | Masked speech + denoising; long the SUPERB front-runner |
| EnCodec | Meta | 2022 | [arXiv 2210.13438](https://arxiv.org/abs/2210.13438) | Neural codec that turns waveforms into discrete tokens for LMs |
| AudioLM | Google | 2022 | [arXiv 2209.03143](https://arxiv.org/abs/2209.03143) | Language modeling over audio tokens; coherent speech/music continuation |
| Whisper | OpenAI | 2022 | [arXiv 2212.04356](https://arxiv.org/abs/2212.04356) | Weakly-supervised multilingual ASR at scale; the default recognizer |
| MusicGen | Meta | 2023 | [arXiv 2306.05284](https://arxiv.org/abs/2306.05284) | Single-stage text-conditioned music generation over codec tokens |
| Stable Audio Open | Stability AI | 2024 | [arXiv 2407.14358](https://arxiv.org/abs/2407.14358) | Open-weight latent-diffusion text-to-audio; 44.1 kHz stereo from CC data |
| F5-TTS | SJTU et al. | 2024 | [arXiv 2410.06885](https://arxiv.org/abs/2410.06885) | Flow-matching DiT zero-shot TTS; no duration model or phoneme alignment |
| Moshi | Kyutai | 2024 | [arXiv 2410.00037](https://arxiv.org/abs/2410.00037) | Full-duplex speech-text model; ~200 ms latency, overlapping live dialogue |
| Qwen2.5-Omni | Alibaba | 2025 | [arXiv 2503.20215](https://arxiv.org/abs/2503.20215) | Thinker-Talker omni model: streaming audio/video in, text + speech out |
| ACE-Step | Team ACE | 2025 | [arXiv 2506.00045](https://arxiv.org/abs/2506.00045) | Open music-generation foundation model with lyric + style alignment |

## Where it stands (2025-2026)
- **Speech-to-speech is the new default.** Moshi (2024) and Qwen2.5-Omni (2025) collapse ASR → LLM → TTS pipelines into one streaming model; full-duplex, sub-300 ms turn-taking is now table stakes.
- **Flow matching beat autoregression for TTS.** Voicebox/F5-TTS-style non-autoregressive DiTs deliver fast, expressive zero-shot cloning without duration models or alignment.
- **Codecs are the bottleneck and the battleground.** Low-bitrate semantic+acoustic codecs (Mimi, descript-style) decide quality, latency, and how cleanly an LM can model audio.
- **Music went open and controllable.** ACE-Step (2025) and ElevenLabs' licensed music model (2025-2026) push genre coverage, lyric alignment, and longer coherence.
- **Convergence into multimodal LLMs.** Audio is increasingly a modality inside an omni-model, not a standalone stack (see 10-multimodal).

## Transferred vs. reinvented
**Transferred directly from language modeling**
- Transformer backbone and next-token decoding over discrete tokens.
- Masked-prediction pretraining and scaling-law intuitions.
- Instruction tuning / RLHF-style alignment for spoken assistants.

**Invented fresh for audio**
- Neural audio tokenization (codecs) and the semantic-vs-acoustic token split.
- Multi-timescale, multi-stream modeling (RVQ token hierarchies, inner-monologue).
- Full-duplex turn-taking and streaming-latency constraints.
- Disentangling content / speaker / prosody for controllable synthesis.

## Open problems
- **Long-range coherence:** minute-scale music and multi-turn conversation still drift in structure and identity.
- **Evaluation:** MOS and WER miss naturalness, expressivity, and dialogue quality; no agreed perceptual benchmark.
- **Safety:** zero-shot voice cloning makes watermarking and consent unsolved at deployment scale.
- **Low-resource & paralinguistics:** most gains concentrate in high-resource languages; emotion, accent, and non-speech audio lag.

## See also
- [01 · Language Foundation Models](../01-language/README.md) — the LM backbone and tokenization ideas audio borrowed.
- [10 · Multimodal Foundation Models](../10-multimodal/README.md) — where omni-models fold speech into a single system.
- [05 · Video Foundation Models](../05-video/README.md) — the other streaming, codec-tokenized modality.
- [Concept Library — Bricks](../../../library/bricks/README.md) — shared primitives: codecs, flow matching, masked prediction.
