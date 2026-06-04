# 10 · Multimodal Foundation Models

*Bind every modality into one token stream — understand and generate across all of them.*

## The adaptation
- **Token:** modality-agnostic. Text BPE, image/audio/video patches as VQ tokens (discrete) or latent vectors (continuous). The frontier trend is one shared sequence holding all of them.
- **Objective:** contrastive alignment (CLIP-style InfoNCE) for retrieval, or unified next-token / next-state prediction — increasingly mixed with a diffusion loss for continuous image/video output.
- **Inductive bias:** a shared embedding space where semantically matched signals (a dog photo, the word "dog", a bark) land near each other; early fusion lets cross-modal attention form from the first layer.
- **Verification:** cross-modal retrieval (R@k), VQA / captioning benchmarks, and — for generators — text-image alignment scores plus human preference (LMArena).

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| CLIP | OpenAI | 2021 | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) | Contrastive image-text pretraining; zero-shot transfer that defined the field |
| Flamingo | DeepMind | 2022 | [arXiv 2204.14198](https://arxiv.org/abs/2204.14198) | Frozen LLM + visual adapter; few-shot in-context VLM |
| LLaVA (Visual Instruction Tuning) | Liu et al. | 2023 | [arXiv 2304.08485](https://arxiv.org/abs/2304.08485) | GPT-4-generated instruction tuning; the open VLM recipe |
| ImageBind | Meta | 2023 | [arXiv 2305.05665](https://arxiv.org/abs/2305.05665) | One space binding 6 modalities via image as the pivot |
| Chameleon | Meta | 2024 | [arXiv 2405.09818](https://arxiv.org/abs/2405.09818) | Early-fusion, token-level interleaving of image and text from scratch |
| Transfusion | Meta | 2024 | [arXiv 2408.11039](https://arxiv.org/abs/2408.11039) | One transformer, two losses: next-token for text + diffusion for images |
| Emu3 | BAAI | 2024 | [arXiv 2409.18869](https://arxiv.org/abs/2409.18869) | "Next-token prediction is all you need" across image, text, video |
| Janus-Pro | DeepSeek | 2025 | [arXiv 2501.17811](https://arxiv.org/abs/2501.17811) | Decouples vision encoders for understanding vs. generation; strong open any-to-any |
| Qwen2.5-Omni | Alibaba | 2025 | [arXiv 2503.20215](https://arxiv.org/abs/2503.20215) | Thinker-Talker omni model; streaming text + speech from any input |
| BAGEL | ByteDance | 2025 | [arXiv 2505.14683](https://arxiv.org/abs/2505.14683) | MoT unified model; reasoning/editing emerge from interleaved pretraining |
| Emu3.5 | BAAI | 2025 | [arXiv 2510.26583](https://arxiv.org/abs/2510.26583) | Native multimodal world model, 10T tokens; DiDA diffusion adaptation for fast decode |
| NExT-OMNI | — | 2025 | [arXiv 2510.13721](https://arxiv.org/abs/2510.13721) | Any-to-any omnimodal via discrete flow matching across text/image/video/audio |

## Where it stands (2025-2026)
- **Native generation won.** Frontier systems (Gemini 2.5 Flash Image, GPT-4o) generate images inside the same architecture that reads them — no bolted-on diffusion module. Conversational editing and multi-image composition follow for free.
- **Unified understanding+generation is the open-model race.** Janus-Pro, BAGEL, and Emu3.5 collapse the perceive/generate split into one model; BAGEL reports reasoning and editing emerging from scale.
- **AR + diffusion hybrids dominate continuous output.** Transfusion's recipe (next-token text, diffusion images) and Emu3.5's DiDA show the field converging on mixed losses over a shared backbone.
- **Omni is now table stakes.** Qwen3-Omni and peers take text+audio+image+video in, stream text+speech out, in real time.
- **Video output is the next cliff.** Late-2026 systems push native video generation into the unified stack, the last modality to fold into any-to-any.

## Transferred vs. reinvented
**Transferred from language modeling**
- Decoder-only transformer + next-token prediction as the universal objective.
- Instruction tuning and RLHF/preference optimization, applied verbatim to VLMs.
- Scaling laws and the "just predict the next token" bet, now across modalities.

**Reinvented for multimodal**
- Tokenizing continuous signals: VQ codebooks vs. continuous latents, and the diffusion loss to decode them.
- Cross-modal alignment objectives (contrastive InfoNCE) with no language analog.
- Fusion design: late (separate encoders → LLM) vs. early (one token stream); decoupled encoders for understand vs. generate.
- Modality-balancing, interleaving curricula, and synchronized position encodings (e.g., TMRoPE).

## Open problems
- **Modality competition.** Adding modalities or balancing the gen/understand loss often regresses one side; clean scaling recipes are unsettled.
- **Continuous vs. discrete tokens.** Discrete VQ is simple but lossy; continuous latents need diffusion and are slow — no clear winner.
- **Grounded, faithful generation.** Compositional prompts, text rendering, and instruction-faithful edits still fail in subtle ways.
- **Evaluation.** Any-to-any models outrun benchmarks; preference arenas are noisy proxies for cross-modal reasoning.

## See also
- [02 · Vision](../02-vision/README.md) — encoders and tokenizers feeding the visual stream
- [04 · Audio & Speech](../04-audio-and-speech/README.md) — the audio modality in omni models
- [05 · Video](../05-video/README.md) — the next modality folding into any-to-any
- [11 · Design Patterns](../11-design-patterns/README.md) — early vs. late fusion, adapters, tokenization
- [Concept library](../../../library/bricks/README.md) — contrastive learning, tokenization, diffusion bricks
