# 30 Papers Worth Knowing Cold

*The papers that built the frontier. Know the key claim, the mechanism, and where it breaks.*

---

## The 10 papers every ML engineer should know

| Paper | Year | The one thing |
|-------|------|--------------|
| Attention Is All You Need (Vaswani et al.) | 2017 | Self-attention + positional encoding = the transformer |
| Adam (Kingma & Ba) | 2014 | Adaptive per-parameter step sizes; default optimizer |
| Deep Residual Learning / ResNet (He et al.) | 2015 | Residual connections → train very deep nets |
| DDPM (Ho et al.) | 2020 | Diffusion models: learn to reverse a noising process |
| Scaling Laws (Kaplan et al.) | 2020 | Loss = power law in compute/data/params |
| GPT-3 (Brown et al.) | 2020 | Scale → in-context learning emerges |
| Chinchilla (Hoffmann et al.) | 2022 | Compute-optimal means training on more data, not just bigger models |
| InstructGPT (Ouyang et al.) | 2022 | RLHF: reward model + PPO → helpful assistant |
| DPO (Rafailov et al.) | 2023 | Preference optimization without RL loop |
| Chain-of-Thought (Wei et al.) | 2022 | Intermediate steps unlock multi-step reasoning |

---

## The 10 papers that define 2024–2026

| Paper | Year | The one thing |
|-------|------|--------------|
| Llama 3 Herd (Dubey et al.) | 2024 | The complete end-to-end frontier recipe |
| DeepSeek-V3 | 2024 | Efficient MoE at frontier scale (~$5.6M training) |
| DeepSeek-R1 | 2025 | Pure RL → reasoning emerges from scratch (R1-Zero) |
| Tülu 3 / RLVR (Lambert et al.) | 2024 | RL with Verifiable Rewards: the post-training recipe |
| Scaling Monosemanticity (Anthropic) | 2024 | SAEs extract millions of interpretable features from real LLMs |
| FlashAttention (Dao et al.) | 2022 | IO-aware attention: O(N) memory, 2–4× faster |
| ZeRO (Rajbhandari et al.) | 2020 | Shard optimizer state → train trillion-parameter models |
| vLLM / PagedAttention (Kwon et al.) | 2023 | KV cache paging → 10–20× higher inference throughput |
| Flow Matching (Lipman et al.) | 2022 | Simulation-free continuous flows; unifying diffusion |
| TabPFN (Hollmann et al.) | 2023 | In-context Bayesian prediction beats tuned GBDTs on tabular data |

---

## The 10 theory/science papers

| Paper | Year | The one thing |
|-------|------|--------------|
| Understanding DL Requires Rethinking Generalization (Zhang et al.) | 2016 | Nets memorize random labels → classical generalization theory fails |
| Grokking (Power et al.) | 2022 | Generalization can arrive long after memorization — a training dynamics anomaly |
| Emergent Abilities (Wei et al.) | 2022 | Capabilities appear discontinuously as model scales |
| Are Emergent Abilities a Mirage? (Schaeffer et al.) | 2023 | Emergence might be a metric artifact — read alongside Wei et al. |
| Toy Models of Superposition (Elhage et al.) | 2022 | Why features superpose: too many features for too few neurons |
| Neural Tangent Kernel (Jacot et al.) | 2018 | Infinite-width nets train like kernel machines — the lazy regime |
| Double Descent (Belkin et al.) | 2019 | Overparameterized models generalize: the U-curve has a second descent |
| Toward Causal Representation Learning (Schölkopf et al.) | 2021 | The agenda for learning latent causal structure |
| Feature Learning Improves Neural Scaling (Bordelon et al.) | 2024 | Why real nets outperform NTK: feature learning, not just lazy regime |
| There Will Be a Scientific Theory of Deep Learning (Simon et al.) | 2026 | The manifesto for a rigorous, predictive DL science |
