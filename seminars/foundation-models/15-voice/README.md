# 15 · Voice Foundation Models

*TTS at scale. Voice cloning. Real-time generation. The speaker-as-context paradigm.*

**The key distinction from audio:** voice FMs focus specifically on human speech synthesis, voice cloning, and real-time TTS — not music or general audio. The "token" is typically a discrete codec token (EnCodec/DAC) or a mel spectrogram patch. The speaker is a conditioning signal, not just content.

**The central challenge:** natural-sounding prosody, emotional expressiveness, and zero-shot voice cloning (generate speech in a new voice from a few seconds of reference audio).

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| VALL-E | Wang et al., Microsoft | 2023 | [arXiv 2301.02111](https://arxiv.org/abs/2301.02111) | LM over EnCodec tokens; zero-shot voice cloning from 3s |
| VALL-E X | Zhang et al., Microsoft | 2023 | [arXiv 2303.03926](https://arxiv.org/abs/2303.03926) | Cross-lingual speech synthesis |
| Voicebox | Le et al., Meta | 2023 | [arXiv 2306.15687](https://arxiv.org/abs/2306.15687) | Flow matching for speech; context-based any-to-any |
| SoundStorm | Borsos et al., Google | 2023 | [arXiv 2305.09636](https://arxiv.org/abs/2305.09636) | Non-autoregressive codec language modeling; fast |
| VoiceCraft | Peng et al. | 2024 | [arXiv 2403.16973](https://arxiv.org/abs/2403.16973) | Zero-shot speech editing and TTS |
| CosyVoice | Du et al., Alibaba | 2024 | [arXiv 2407.05407](https://arxiv.org/abs/2407.05407) | Multilingual zero-shot TTS; flow matching |
| F5-TTS | Chen et al. | 2024 | [arXiv 2410.06885](https://arxiv.org/abs/2410.06885) | Diffusion TTS with flow matching; natural prosody |
| Tortoise TTS | Betker, J. | 2023 | [GitHub](https://github.com/neonbjb/tortoise-tts) | High-quality multi-voice TTS; widely used open model |
| Fish Speech | Fish Audio | 2024 | [GitHub](https://github.com/fishaudio/fish-speech) | Fast open-source TTS with voice cloning |

**The real-time constraint:** the latest direction is sub-100ms latency voice generation for live conversation AI — streaming codec token generation with minimal buffering. This adds engineering constraints (streaming transformer architectures) on top of the quality requirements.
**The ethical dimension:** voice cloning from seconds of audio raises consent questions that are more acute than most other FM domains.
