# 15 · Voice Foundation Models
*Speech as a token stream — clone a voice from seconds, then speak in real time.*

## The adaptation
- **Token:** discrete acoustic/semantic codes from a neural codec (EnCodec, DAC, Mimi) or supervised semantic tokens; some flow-matching models predict continuous mel frames instead.
- **Objective:** next-token prediction over codec codes (autoregressive) or mask-and-predict / flow matching (non-autoregressive parallel decoding).
- **Inductive bias:** speaker identity is a *conditioning prompt* (a few seconds of reference audio), separate from linguistic content; many models split a coarse semantic stream from fine acoustic detail.
- **Verification:** WER via ASR transcription (intelligibility), speaker-similarity cosine (cloning fidelity), and human MOS / Elo arenas (naturalness, prosody, emotion).

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| VALL-E | Microsoft | 2023 | [arXiv 2301.02111](https://arxiv.org/abs/2301.02111) | First "TTS as LM": zero-shot voice cloning from a 3s prompt; reframed the field |
| Voicebox | Meta | 2023 | [arXiv 2306.15687](https://arxiv.org/abs/2306.15687) | Flow matching for speech; non-autoregressive any-to-any infilling |
| MaskGCT | Amphion | 2024 | [arXiv 2409.00750](https://arxiv.org/abs/2409.00750) | Fully NAR masked codec transformer; no alignment or duration model needed |
| Moshi | Kyutai | 2024 | [arXiv 2410.00037](https://arxiv.org/abs/2410.00037) | Full-duplex speech-to-speech dialogue; ~200ms latency; "inner monologue" text prefix |
| F5-TTS | SJTU | 2024 | [arXiv 2410.06885](https://arxiv.org/abs/2410.06885) | Flow-matching diffusion TTS; simple, fast, strong open baseline |
| Sesame CSM-1B | Sesame | 2025 | [HF sesame/csm-1b](https://huggingface.co/sesame/csm-1b) | Llama backbone + Mimi codec; context-aware conversational expressiveness |
| CosyVoice 3 | Alibaba | 2025 | [arXiv 2505.17589](https://arxiv.org/abs/2505.17589) | Scaled to 1M hours / 1.5B params; differentiable reward post-training; in-the-wild |
| IndexTTS2 | Bilibili | 2025 | [arXiv 2506.21619](https://arxiv.org/abs/2506.21619) | Disentangles emotion from timbre; explicit duration control; AAAI |
| VoXtream | KTH | 2025 | [arXiv 2509.15969](https://arxiv.org/abs/2509.15969) | Full-stream AR TTS; ~102ms first-packet latency on GPU |
| Qwen3-TTS | Alibaba | 2026 | [GitHub](https://github.com/QwenLM/Qwen3-TTS) | Open 0.6B/1.7B; 10 languages, voice design + 3s clone; tops ElevenLabs/MiniMax on WER |

## Where it stands (2025-2026)
- **Open weights caught up to closed APIs.** Qwen3-TTS (Jan 2026) and Fish Audio S2 lead Elo/TTS-Arena boards, beating ElevenLabs and MiniMax on WER and speaker similarity.
- **Real-time is solved at the architecture level.** Full-stream models (VoXtream, SpeakStream, SyncSpeech) push first-packet latency near 100ms; full-duplex S2S (Moshi) handles overlapping turns.
- **Control is the new frontier:** independent control of emotion, duration, and timbre (IndexTTS2), plus natural-language "voice design" prompts (Qwen3-TTS).
- **Codec convergence:** Mimi-style streaming codecs and supervised semantic tokenizers are displacing raw EnCodec for low-latency, high-naturalness pipelines.

## Transferred vs. reinvented
**Transferred from language modeling**
- Decoder-only transformer + next-token prediction over discrete codes.
- Scaling laws: 1M-hour / 1.5B-param runs (CosyVoice 3) buy naturalness and robustness.
- In-context conditioning: the reference voice is just a prompt prefix.
- RLHF-style post-training via differentiable reward models.

**Reinvented for voice**
- Neural codec tokenization and semantic/acoustic stream splitting.
- Flow-matching / masked-generative NAR decoders for parallel, controllable synthesis.
- Streaming, sentence-length-independent latency and full-duplex turn-taking.
- Disentanglement of timbre, emotion, and duration as separate controllable axes.

## Open problems
- **Robustness:** AR models still hallucinate, repeat, or drop words on hard text and code-switching.
- **Long-form & multi-speaker consistency:** stable timbre and prosody across minutes of dialogue.
- **Evaluation:** MOS saturates; WER and similarity miss prosody, emotion, and naturalness — Elo arenas are a stopgap.
- **Consent & misuse:** cloning from seconds of audio outpaces watermarking and detection.

## See also
- [04 · Audio & Speech Foundation Models](../04-audio-and-speech/README.md) — ASR, music, general audio; voice's parent domain.
- [01 · Language Foundation Models](../01-language/README.md) — the LM backbone voice models borrow.
- [10 · Multimodal Foundation Models](../10-multimodal/README.md) — speech-to-speech and voice-enabled assistants.
- [Concept library · Bricks](../../../library/bricks/README.md) — codecs, flow matching, RLHF, scaling laws.
