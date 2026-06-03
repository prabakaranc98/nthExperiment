# How Frontier Models Are Built — 100-Day Mastery Curriculum

*The path through the frontier-models survey. ~100 days, 9 phases. Each phase = an objective, daily targets, core primary sources, and **one build milestone**. Pairs with: `frontier-model-survey.md` (the map), `30-builds-roadmap.md` (the shipping plan).*

**The one rule:** every phase ships a runnable artifact to a public repo before moving on. *First commit before first paper.*
**Pacing:** ~1 unit/day. Compress fundamentals you already have; spend saved days on the three high-leverage phases — RL/post-training, systems/kernels, interpretability.

---

## Phase 0 — Orientation & the canonical recipe (Days 1–4)

**Objective:** Get the whole pipeline in your head as one object before zooming in.

| Day | Target |
|---|---|
| 1 | Read the survey's "seven shifts" + skim every deep profile. Build a one-page mental model: data → pretrain → post-train → align/eval → serve. |
| 2 | **Llama 3 Herd** ([2407.21783](https://arxiv.org/abs/2407.21783)) — read once, fast, for *shape*. |
| 3 | Re-read Llama 3's data + scaling sections slowly. Note every design decision and *why*. |
| 4 | Scaling laws: Kaplan ([2001.08361](https://arxiv.org/abs/2001.08361)) → Chinchilla ([2203.15556](https://arxiv.org/abs/2203.15556)). Compute-optimal vs. inference-optimal (over-training). |

**Build:** Repo skeleton + a "frontier model build" diagram (data→serve) you annotate all 100 days.

---

## Phase 1 — Statistical & data foundations (Days 5–18)

**Objective:** The probabilistic and data-centric base. *Frontier models are 90% a data problem.*

| Days | Block | Targets |
|---|---|---|
| 5–7 | **Probabilistic core** | MLE as cross-entropy / next-token prediction; KL, entropy, perplexity, MI; LM training as density estimation; Bayesian vs. frequentist generalization |
| 8–9 | **Tokenization** | BPE, byte-level BPE, SentencePiece/Unigram; vocab-size tradeoffs; implement BPE from scratch |
| 10–12 | **Data curation** | Crawl → filter → quality classifiers → dedup (MinHash/LSH); RefinedWeb/FineWeb; DoReMi-style data-mixing |
| 13–14 | **Synthetic data** | Phi-4 thesis ([2412.08905](https://arxiv.org/abs/2412.08905)); distillation as data generation; collapse/contamination risks |
| 15–16 | **Scaling & data laws** | Data-constrained scaling, repetition value, token-efficiency |
| 17–18 | **Decontamination & eval hygiene** | n-gram/fuzzy decontamination (Claude Opus 4.5 card procedure). An epistemics problem, not a checkbox. |

**Build:** Mini data pipeline — sample a corpus, dedup, train a tokenizer, compute a mixing weight, produce a dataset card.

---

## Phase 2 — Architecture & efficiency (Days 19–36)

**Objective:** Every architectural lever the frontier pulls and the efficiency reason behind each.

| Days | Block | Targets |
|---|---|---|
| 19–21 | **Transformer internals** | Attention Is All You Need ([1706.03762](https://arxiv.org/abs/1706.03762)); pre/post-norm, RMSNorm, residual stream as communication channel. Implement a block from scratch. |
| 22–23 | **Attention variants** | MHA → MQA → GQA → **MLA**; the KV cache as the long-context bottleneck |
| 24–25 | **Positional encoding** | RoPE, ALiBi, NoPE; long-context extension (YaRN, position interpolation) |
| 26–29 | **Mixture-of-Experts** | DeepSeekMoE, aux-loss-free balancing, MTP ([2412.19437](https://arxiv.org/abs/2412.19437)); Mixtral ([2401.04088](https://arxiv.org/abs/2401.04088)); implement toy MoE layer + router |
| 30–32 | **Beyond pure transformers** | SSMs / Mamba ([2312.00752](https://arxiv.org/abs/2312.00752)); linear attention; hybrids (Qwen3-Next Gated Delta Net) |
| 33–34 | **Multimodal** | Early-fusion (Llama 4) vs. adapter/encoder; vision encoders; omni models |
| 35–36 | **Numerical efficiency** | FP8/FP4 training & quantization (QAT, GPTQ, AWQ); speculative decoding; precision/stability frontier |

**Build:** From-scratch decoder-only LM (RMSNorm + RoPE + GQA) trained on Phase-1 corpus. Add MoE variant; measure active-vs-total-param tradeoff.

---

## Phase 3 — Infra, systems & kernels (Days 37–52)

**Objective:** How frontier-scale training/serving physically happens — the layer most researchers skip and most research-engineer interviews probe.

| Days | Block | Targets |
|---|---|---|
| 37–38 | **GPU model** | Memory hierarchy, SM/warp execution, roofline, arithmetic intensity; why everything is memory-bound |
| 39–41 | **Kernels** | CUDA → Triton; write a fused kernel. FlashAttention 1/2/3 ([2205.14135](https://arxiv.org/abs/2205.14135), [2307.08691](https://arxiv.org/abs/2307.08691)) — IO-awareness as the core idea |
| 42–45 | **Distributed training** | Data parallel → ZeRO/FSDP ([1910.02054](https://arxiv.org/abs/1910.02054)); tensor + pipeline (Megatron-LM, [1909.08053](https://arxiv.org/abs/1909.08053)); 3D/4D; NCCL, DualPipe |
| 46–47 | **Training stability** | Loss spikes, clipping, init/LR, μP; MuonClip/QK-clip (Kimi's zero-spike 15.5T run); FP8 stability |
| 48–49 | **Hardware-aware co-design** | DeepSeek-V3 hardware paper; how cluster topology shapes architecture |
| 50–52 | **Inference & serving** | KV-cache management, PagedAttention/vLLM ([2309.06180](https://arxiv.org/abs/2309.06180)), continuous batching, prefill/decode disaggregation, quantized serving |

**Build:** Take Phase-2 model multi-GPU (FSDP), write one Triton kernel, stand up a minimal paged-KV inference server. Benchmark tokens/sec and cost.

---

## Phase 4 — Post-training & RL (Days 53–72) — *the deep RL block*

**Objective:** The full post-training stack with reasoning RL as centerpiece — the most important phase for understanding 2024–2026.

| Days | Block | Targets |
|---|---|---|
| 53–54 | **SFT & instruction tuning** | Data construction, chat templates, packing; Tülu 3 open recipe ([2411.15124](https://arxiv.org/abs/2411.15124)) |
| 55–56 | **Parameter-efficient FT** | LoRA/QLoRA ([2106.09685](https://arxiv.org/abs/2106.09685)); when full FT is worth it |
| 57–59 | **Classic RLHF** | InstructGPT ([2203.02155](https://arxiv.org/abs/2203.02155)); reward modeling; PPO for LMs; KL-control; Goodhart's law |
| 60–61 | **Offline preference optimization** | DPO ([2305.18290](https://arxiv.org/abs/2305.18290)) + variants (IPO, KTO, ORPO, SimPO); when offline beats online |
| 62–64 | **RLVR + GRPO** | RLVR (Tülu 3); GRPO (DeepSeekMath → R1) — critic-free, group-relative advantage. Implement GRPO on a toy verifiable task. |
| 65–67 | **Reasoning models** | R1 + Nature version ([2501.12948](https://arxiv.org/abs/2501.12948)): R1-Zero, cold-start, multi-stage; inference-time scaling (o1); process vs. outcome rewards |
| 68–70 | **Agentic & tool-use post-training** | Tool-use RL, agentic-trajectory synthesis (Kimi K2, [2507.20534](https://arxiv.org/abs/2507.20534)); compaction; Grok 4 RL-at-scale |
| 71–72 | **Distillation & specialization** | Reasoning-trace distillation, co-distillation (Llama 4 Behemoth → Scout/Maverick) |

**Build:** Post-train Phase-2 base: SFT → DPO → small GRPO run on a verifiable reward. Show the reasoning-length-grows curve on a held-out task.

---

## Phase 5 — Behavior, alignment, safety, evals & interpretability (Days 73–84)

**Objective:** The half of the pipeline that isn't capability — where safety-focused labs hire.

| Days | Block | Targets |
|---|---|---|
| 73–74 | **Alignment methods** | Constitutional AI/RLAIF ([2212.08073](https://arxiv.org/abs/2212.08073)); AI vs. human feedback; deliberative alignment |
| 75–76 | **Evals science** | Benchmark design, contamination, saturation, held-out "production" benchmarks; pass@k, arenas, LLM-as-judge pitfalls |
| 77–78 | **Dangerous-capability frameworks** | RSP/ASL, Preparedness, Frontier Safety. Read one frontier system card in full ([Claude Opus 4.5](https://www.anthropic.com/claude-opus-4-5-system-card) or [GPT-5](https://cdn.openai.com/gpt-5-system-card.pdf)) and map its structure. |
| 79–80 | **Misalignment phenomena** | Reward hacking, sabotage, scheming/deception, evaluation awareness, sycophancy; CoT monitorability |
| 81–84 | **Mechanistic interpretability** | Induction heads & ICL; Toy Models of Superposition; Scaling Monosemanticity (SAEs at production scale); circuits, activation patching, probing. Train a small SAE on your model's activations. |

**Build:** Small eval + interp harness — LLM-as-judge suite, reward-hacking probe, SAE extracting a few interpretable features.

---

## Phase 6 — The science & theory of LLMs (Days 85–92)

**Objective:** *Why* any of this works — the conceptual layer separating an engineer who tunes from a researcher who explains.

| Days | Block | Targets |
|---|---|---|
| 85 | **Emergence debate** | "Emergent Abilities" ([2206.07682](https://arxiv.org/abs/2206.07682)) vs. "Mirage" ([2304.15004](https://arxiv.org/abs/2304.15004)) |
| 86 | **Grokking** | [2201.02177](https://arxiv.org/abs/2201.02177) — delayed generalization, circuit formation |
| 87 | **ICL theory** | ICL as implicit gradient descent / Bayesian inference; induction heads as mechanism |
| 88–89 | **Physics of LLMs** | Allen-Zhu's "Physics of Language Models" series — controlled experiments on knowledge storage and reasoning |
| 90 | **Optimization & loss landscapes** | Why SGD/Adam generalize; double descent; flat minima; edge of stability |
| 91 | **Why depth works** | NTK vs. feature learning; representation learning; lottery ticket / pruning |
| 92 | **Scaling theory** | Mechanistic explanations of scaling-law exponents; what the numbers mean |

**Build:** Reproduce grokking on modular arithmetic, identify the induction-head-like circuit, write it up as a short public post.

---

## Phase 7 — The scientific-application frontier (Days 93–97)

**Objective:** Where frontier methods meet real science — and the bridge to your CRL/world-models/MARL research.

| Days | Block | Targets |
|---|---|---|
| 93 | **Math reasoning** | AlphaProof ([Nature](https://www.nature.com/articles/s41586-025-09833-y)): RL in Lean, auto-formalization; Gemini Deep Think IMO-gold |
| 94 | **Life science I** | AlphaFold 3 (Nature 2024) — diffusion over biomolecular complexes |
| 95 | **Life science II** | ESM3 ([Science](https://www.science.org/doi/10.1126/science.ads0018)) — multimodal generative protein LM |
| 96 | **Physics / materials** | GNoME + MatterGen ([2312.03687](https://arxiv.org/abs/2312.03687)) — inverse design of crystals |
| 97 | **World models & your thread** | Connect world-model / agent work to your CRL × continual-learning and world-models × MARL bets |

**Build:** Pick one domain — reproduce a small piece. Write the method up against its primary paper.

---

## Phase 8 — Company audit (Days 98–99)

**Objective:** Synthesize the landscape lab-by-lab — a memory aid and interview asset.

| Lab | Signature methods |
|---|---|
| **DeepSeek** | MLA, DeepSeekMoE, aux-loss-free balancing, MTP, FP8/DualPipe, GRPO, pure-RL reasoning |
| **Meta** | Complete end-to-end recipe (Llama 3); MoE + early-fusion multimodal + 10M context (Llama 4); co-distillation |
| **Alibaba (Qwen)** | Breadth (0.6B–235B), unified thinking/non-thinking, thinking budgets, extreme sparsity (Qwen3-Next) |
| **Moonshot (Kimi)** | MuonClip, trillion-scale zero-spike training, agentic-first post-training |
| **AI2** | Full openness (OLMo 2 + Tülu 3); RLVR formalization. Reproducibility anchor. |
| **OpenAI** | Inference-time scaling (o1), reasoning RL, router systems (GPT-5), Preparedness |
| **Anthropic** | Constitutional AI/RLAIF, hybrid reasoning, RSP/ASL, SAE interpretability, CoT-no-train stance |
| **Google DeepMind** | Long context (Gemini 1.5), Deep Think parallel reasoning, Frontier Safety; AlphaFold/AlphaProof/GNoME |
| **xAI** | RL at pretraining-scale compute (Grok 4, Colossus) |
| **Microsoft / NVIDIA / Mistral / Cohere / IBM / Zhipu** | Synthetic data (Phi), reward models (Nemotron), Western MoE (Mixtral), RAG/tool-use (Command R), enterprise (Granite), bilingual agentic (GLM) |

**Build:** One-page-per-lab audit doc (lineage + methods + posture). Directly useful for research-engineer interviews.

---

## Phase 9 — Capstone (Days 100 → ongoing)

Take a small model (≈100M–1B) the whole way:

1. **Data** — curate + dedup + mix + tokenize (Phase 1)
2. **Architecture** — decoder-only with RoPE + GQA, optional MoE (Phase 2)
3. **Systems** — multi-GPU FSDP, one custom kernel, logged stability (Phase 3)
4. **Post-train** — SFT → DPO → GRPO on a verifiable reward (Phase 4)
5. **Eval + interp** — eval harness, reward-hacking probe, small SAE (Phase 5)
6. **Serve** — paged-KV inference server with throughput/cost benchmarks (Phase 3)
7. **Write-up** — a technical report in the style of the ones you've been reading: architecture, data, training, evals, safety notes, limitations

That report is the deliverable. Because you built one end to end.

---

## A realistic note on scope

Protect the three high-leverage phases (RL/post-training, systems/kernels, interpretability) and move fast through fundamentals you already have. For any phase, pull current papers right before you start it — the frontier moves monthly. The builds, not the reading, are what convert this into mastery.
