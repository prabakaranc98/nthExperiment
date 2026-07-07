# 30 Builds — A 6-Month Frontier-AI Build Roadmap

*Converting the survey + curriculum into shipped work. 30 serious builds over ~26 weeks. One continuous thread (**nanoLM**, taken end-to-end) with weekly **bricks** orbiting a monthly **arc**. Every build has a hard "ships when…" line — that line is the build.*

**FAIRE context:** **one route** across the terrain — a scaffold for orientation and blind-spot awareness, **not a schedule**. The real operating unit is a [~100-hour block on one module, driven by a live question](../now.md) ([a living system, not a checklist](../faire-program.md#how-this-actually-runs--a-living-system)). Use this map to see what you're *skipping*, not to pace yourself — and let builds emerge from questions, never from a résumé/JD. → [Program Handbook](../faire-program.md) · [Now](../now.md)

---

## The cadence

**Each month = 1 arc (~3–4 weeks) + 4 bricks (~1 week each) = 5 builds. × 6 months = 30.**

- **Brick** — sharp, self-contained, from-scratch, ~1 week. Standalone-impressive.
- **Arc** — integrative, ~2–4 weeks, runs *alongside* bricks and absorbs them.
- **The thread:** builds compose. nanoLM gets pretrained (M1), parallelized (M2), served (M3), reasoning-RL'd (M4), interpreted/aligned (M5), shipped as a full technical report (M6). By month 6 the capstone is assembly, not a cold start.
- **Ships publicly:** repo + a short writeup on your proof-of-work surface, every time. No private half-builds.

---

## Three rules that protect shipping

1. **The "ships when" line is non-negotiable scope.** Hit it, push, move on. A 70% build that ships beats a 95% build that doesn't.
2. **No new framework before the current build ships.** Tooling/OS work doesn't count as a build.
3. **One arc at a time.** Bricks can parallelize; arcs cannot.

---

## Month 1 — Data → Transformer core
*Draws on curriculum Phases 1–2. The base of nanoLM.*

| # | Build | Type | Ships when… |
|---|---|---|---|
| 1 | **BPE tokenizer from scratch** | Brick | Trained on multilingual+code+math corpus; vocab-size sweep with perplexity/compression table |
| 2 | **Data pipeline** | Brick | MinHash dedup + quality filter + computed data-mixing weights → a documented dataset card |
| 3 | **Decoder-only transformer from scratch** | Brick | RMSNorm + RoPE + GQA; trains a tiny LM with a clean loss curve and a sampling demo |
| 4 | **Attention-variant bench** | Brick | MHA vs MQA vs GQA vs MLA — KV-cache memory + quality plotted side by side |
| 5 | **nanoLM-base** | **Arc** | A clean pretraining repo combining 1–4, trained on your corpus, with a model card + reproducible config |

---

## Month 2 — Architecture depth → Systems & kernels
*Draws on Phases 2–3. Make nanoLM efficient and scalable.*

| # | Build | Type | Ships when… |
|---|---|---|---|
| 6 | **MoE layer from scratch** | Brick | Router + fine-grained + shared experts + aux-loss-free balancing; active/total-param tradeoff measured |
| 7 | **Mamba / SSM block** | Brick | From scratch; benchmarked vs attention on long sequences (throughput + quality) |
| 8 | **Triton fused kernel** | Brick | A fused op (e.g., RMSNorm or fused softmax) benchmarked vs PyTorch eager |
| 9 | **IO-aware attention** | Brick | Mini FlashAttention (tiling + online softmax); correctness + memory-vs-naive plot |
| 10 | **Multi-GPU nanoLM** | **Arc** | FSDP training + custom kernel + logged stability (grad norms, spike handling) + throughput/MFU report |

---

## Month 3 — Inference/serving → Post-training start
*Draws on Phases 3–4. Make nanoLM servable, then begin shaping it.*

| # | Build | Type | Ships when… |
|---|---|---|---|
| 11 | **Paged-KV inference server** | Arc-lite | Mini-vLLM with continuous batching; throughput/latency/cost benchmark vs naive generate |
| 12 | **Speculative decoding** | Brick | Draft+verify implemented; measured speedup at fixed quality |
| 13 | **SFT pipeline** | Brick | Chat template + packing + instruction data; nanoLM-base → nanoLM-instruct with before/after samples |
| 14 | **LoRA / QLoRA from scratch** | Brick | Adapter math implemented; quality + memory vs full FT compared |
| 15 | **Quantized serving** | **Arc** | FP8/INT4 (GPTQ or AWQ) quantized nanoLM served; quality-vs-throughput tradeoff report |

---

## Month 4 — RL / reasoning *(the deep block)*
*Draws on Phase 4. The most important month — the 2024–26 recipe.*

| # | Build | Type | Ships when… |
|---|---|---|---|
| 16 | **Reward model + PPO** | Arc-lite | Classic RLHF on a toy preference task; reward-over-optimization (Goodhart) curve demonstrated |
| 17 | **DPO + variants** | Brick | DPO from scratch; DPO vs KTO vs SimPO compared on the same data |
| 18 | **GRPO from scratch** | **Arc** | Group-relative advantage on a verifiable task; the reasoning-length-grows curve reproduced. *(Centerpiece.)* |
| 19 | **Verifiable-reward environment** | Brick | A math/code-test RLVR env with a clean reward API + a short training run on it |
| 20 | **Mini-R1** | **Arc** | nanoLM-base → SFT cold-start → GRPO reasoning RL → distilled to a smaller model. End-to-end reasoning recipe, written up. |

*(M4 runs two arcs by design — 18 feeds 20. If time-boxed, 18 is the must-ship; 20 can roll into M5.)*

---

## Month 5 — Safety / evals / interpretability
*Draws on Phase 5. The half that isn't capability — where safety-focused labs hire.*

| # | Build | Type | Ships when… |
|---|---|---|---|
| 21 | **Eval harness** | Brick | pass@k + LLM-as-judge + contamination/decontamination check, run on nanoLM |
| 22 | **Reward-hacking probe suite** | Brick | Impossible-task + hidden-test hacking detection; nanoLM's hack-rate reported |
| 23 | **Sparse autoencoder** | **Arc** | SAE trained on nanoLM activations; a handful of interpretable features surfaced + visualized |
| 24 | **Circuit analysis** | Brick | Activation patching to locate induction heads; one circuit written up |
| 25 | **Constitutional-AI mini-loop** | **Arc** | AI-feedback (RLAIF) alignment on nanoLM + before/after behavior eval + a mini "system card" |

---

## Month 6 — Science frontier → Capstone → your research bets
*Draws on Phases 6–7 + 9. Tie the pipeline to theory, science, and your own thread.*

| # | Build | Type | Ships when… |
|---|---|---|---|
| 26 | **Reproduce grokking** | Brick | Modular-arithmetic grokking + identified circuit-formation transition; short theory post |
| 27 | **Science-domain reproduction** | **Arc** | Pick one: fine-tune ESM2/3 on a motif task / run+analyze MatterGen or GNoME / a small Lean prover. Written up vs its primary paper. |
| 28 | **Research-bet bridge** | Arc-lite | A small experiment tied to your CRL×continual-learning or world-models×MARL bet (e.g., JEPA+MPC or PRPO probe) |
| 29 | **Capstone tech report** | **Arc** | The full nanoLM pipeline (data→pretrain→post-train→eval→safety→serve) written as a frontier-style technical report |
| 30 | **Portfolio synthesis** | Brick | The 30-build index + company-audit one-pager + a "what I learned building a frontier model end-to-end" post |

---

## What you have at the end

- **One coherent thread:** nanoLM taken from raw text to a served, reasoning-capable, interpreted, aligned, documented model — the rare portfolio piece showing the *whole* pipeline
- **~24 standalone bricks** — each a credible "I implemented X from scratch" interview artifact
- **6 arc-builds** that read as project-level work
- **A capstone technical report** in the genre you've been studying

---

## If a month slips

Protect the arc (the **bold** rows: 5, 10, 15, 18/20, 23/25, 29) and the GRPO centerpiece (18). Bricks are sacrificeable; the thread is not. A finished thread with 22 builds beats a broken thread with 30.

---

## Tracker stub

```
M_ / Build _ : <name>
Ships when: <the line>
Status: [ ] not started  [ ] in progress  [ ] SHIPPED → <link>
```
