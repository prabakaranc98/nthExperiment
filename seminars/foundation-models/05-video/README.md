# 05 · Video Foundation Models

*Add time to vision and the model starts simulating physics — is a great video model secretly a world model?*

## The adaptation
- **Token:** a spatiotemporal patch — a small space-time cube of pixels (or its latent code from a learned video tokenizer).
- **Objective:** split into two camps — *generative* (denoise/predict pixels via diffusion transformers; the dominant text-to-video recipe) and *latent-predictive* (JEPA-style masked feature prediction; no pixel reconstruction).
- **Inductive bias:** factorized space-time attention, causal masking over time, and **action conditioning** to turn a passive generator into a controllable simulator.
- **Verification signal:** does it obey physics, object permanence, and 3D consistency under interaction — not just look sharp? The live question is generalization vs. interpolation.

## Landmark → Frontier

| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| VideoGPT | Yan et al. | 2021 | [arXiv 2104.10157](https://arxiv.org/abs/2104.10157) | VQ-VAE + autoregressive transformer; early proof video tokens are learnable. |
| GAIA-1 | Wayve | 2023 | [arXiv 2309.17080](https://arxiv.org/abs/2309.17080) | Action-conditioned driving world model — video as a controllable simulator. |
| Genie | DeepMind | 2024 | [arXiv 2402.15391](https://arxiv.org/abs/2402.15391) | Learned *latent actions* from unlabeled video; playable 2D worlds with no action labels. |
| Sora (Video Generation as World Simulators) | OpenAI | 2024 | [Technical report](https://openai.com/research/video-generation-models-as-world-simulators) | Diffusion transformer at scale; set the "world simulator" framing for the field. |
| CogVideoX | THUDM | 2024 | [arXiv 2408.06072](https://arxiv.org/abs/2408.06072) | First strong *open* DiT text-to-video; 3D-VAE + expert transformer. |
| V-JEPA | Bardes et al., Meta | 2024 | [arXiv 2404.08471](https://arxiv.org/abs/2404.08471) | Latent masked prediction — understanding without generating a single pixel. |
| NVIDIA Cosmos WFM | NVIDIA | 2025 | [arXiv 2501.03575](https://arxiv.org/abs/2501.03575) | Open world-foundation-model platform (tokenizers + diffusion/AR) for physical AI. |
| Wan (open video) | Alibaba Wan Team | 2025 | [arXiv 2503.20314](https://arxiv.org/abs/2503.20314) | Strongest open T2V/I2V base; MoE diffusion in Wan2.2 (27B total, 14B active). |
| V-JEPA 2 | Meta | 2025 | [arXiv 2506.09985](https://arxiv.org/abs/2506.09985) | 1M+ hrs pretraining; action-conditioned post-train enables **zero-shot robot planning**. |
| Genie 3 | DeepMind | 2025 | [Blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | First **real-time interactive** world model: 720p/24fps, ~1 min consistency. |
| Cosmos World Simulation (Physical AI) | NVIDIA | 2025 | [arXiv 2511.00062](https://arxiv.org/abs/2511.00062) | Video FMs explicitly framed and benchmarked as world simulators for robotics. |

## Where it stands (2025-2026)
- **Generation has saturated fidelity, racing on control & audio.** Native joint audio-video (Veo 3, ByteDance Seedance 2.0, Kling 3.0), longer clips, and 4K are now table stakes; quality leadership flips monthly on public Elo boards.
- **Interactivity is the new frontier.** Genie 3 made worlds you *steer* in real time; the bet is shifting from "pretty clips" to "playable, persistent environments."
- **World models went operational.** Genie 3 powers Waymo's driving simulator (2026); Cosmos and V-JEPA 2 close the loop into robot policy training and planning.
- **Two paradigms diverge.** Pixel diffusion (Sora/Veo/Wan) wins generation; latent-predictive JEPA wins efficient understanding and planning (V-JEPA 2 plans ~15x faster than a Cosmos-based baseline).
- **Open vs. closed gap narrowed.** Wan and Cosmos put near-frontier weights in the open; the remaining moat is data scale and real-time interaction.

## Transferred vs. reinvented
**Transferred from language modeling**
- Tokenize → transformer → scale; the next-token / denoising-prediction loop.
- Self-supervised masked prediction (V-JEPA inherits the MAE/BERT recipe in latents).
- Scaling laws and emergent capability with data + compute.

**Reinvented for video**
- Spatiotemporal patch tokenizers and learned 3D-VAE latents.
- Factorized space-time attention and causal temporal masking.
- **Action conditioning and *learned* latent actions** (Genie) — the bridge from generator to controllable simulator.
- Diffusion-transformer hybrids as the default generative backbone (vs. pure AR in text).

## Open problems
- **Physics, not vibes:** persistent object permanence, contact dynamics, and long-horizon consistency still break under interaction.
- **World model or pattern matcher?** No clean benchmark separates genuine causal/counterfactual reasoning from high-fidelity interpolation.
- **Compute:** real-time, multi-minute interactive generation is brutally expensive; memory beyond ~1 minute degrades.
- **Evaluation:** fidelity metrics (FVD) reward look, not correctness or controllability — closed-loop, task-grounded eval is immature.

## See also
- [../02-vision/README.md](../02-vision/README.md) — spatial backbone video extends into time.
- [../10-multimodal/README.md](../10-multimodal/README.md) — text/audio conditioning and native joint A/V.
- [../15-voice/README.md](../15-voice/README.md) — the audio half of modern video models.
- [../../../library/bricks/README.md](../../../library/bricks/README.md) — diffusion, tokenizers, JEPA, scaling-laws primitives.
