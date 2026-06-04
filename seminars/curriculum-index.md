# Frontiers in AI & Research Engineering — Curriculum Index

*The index that ties the whole library together: one navigable curriculum spanning frontier models, AI, LLMs, ML, DL, data science, applied statistics, and systems/infra engineering. Built on the principle that you stand at the frontier first, and the gaps drag you down to the immortal fundamentals.*

---

## Part A — The library

Seven documents. Four are **maps** (the terrain); two are the **path** (execution); this index is the seventh — the table of contents and the route planner.

| # | Doc | Role | Type |
|---|---|---|---|
| 1 | [Frontiers in Machine Learning](frontiers-ml.md) | *what learning is* — paradigms, theory, RL, causal, probabilistic | Map |
| 2 | [Frontiers in Deep Learning](frontiers-dl.md) | *neural networks specifically* — architectures, diffusion/flow, DL theory, interpretability, neural operators | Map |
| 3 | [Frontiers in Data Science](frontiers-ds.md) | *reliable knowledge & decisions* — tabular/TS FMs, causal at scale, conformal, agentic | Map |
| 4 | [Applied Statistics for the Modern AI Era](applied-statistics-ai-era.md) | *the inferential bedrock* — evals, calibration, conformal, PPI, anytime-valid | Map |
| 5 | [Frontier Models Survey](frontier-model-survey.md) | how frontier models specifically are built — the map of the last 2 years | Map |
| 6 | [100-Day Mastery Curriculum](frontier-models-100day-curriculum.md) | the path through the survey, end to end | Path |
| 7 | [30-Build Roadmap](30-builds-roadmap.md) | 6-month shipping plan: 30 builds, one model taken end-to-end | Path |

Also: [Frontier of ML/DL/DS — Wide-Angle Map](frontier-ml-dl-ds-map.md) | [How to Read a Paper](how-to-read.md)

**Deep-dive course:** [APSL v3 — Algorithmic Probabilistic Structure Learning](apsl-structure-learning.md) — an 18-week graduate syllabus threading P1 (probabilistic foundations), P4 (causal ML), and the amortized-inference frontier (GFlowNets · PFNs · foundation models as structure learners).

---

## Part B — The curriculum (eight pillars)

| Pillar | Covers | Maps to | Frontier weight |
|---|---|---|---|
| **P0 · Mathematical & statistical foundations** | Linear algebra, probability, optimization, **statistical inference & causal foundations** | Doc 4 + core | Immortal — never skip |
| **P1 · ML foundations & frontier** | Learning paradigms, generalization theory, in-context learning, causal ML | Doc 1 | High |
| **P2 · DL foundations & frontier** | Transformers→SSMs, **diffusion→flow matching**, DL theory, **interpretability** | Doc 2 | High |
| **P3 · LLMs & frontier models** | The full pretrain→serve pipeline, MoE, scaling, data | Doc 5, 6 | **Highest** |
| **P4 · RL & post-training** | RLHF, DPO, **RLVR/GRPO, reasoning RL**, agentic | Docs 1, 5 | **Highest** |
| **P5 · Interpretability, alignment & safety** | SAEs, circuits, CAI, reward hacking, eval/CoT monitorability | Docs 2, 5 | **Highest (esp. Anthropic)** |
| **P6 · Systems & infra engineering** | **Kernels (Triton/CUDA), distributed training, serving, FP8** | Doc 5 (systems) | **Highest (RE-defining)** |
| **P7 · Data science & applied stats for AI** | **Eval statistics, PPI, conformal, calibration**, causal inference, tabular/TS FMs | Docs 3, 4 | High & differentiating |

**Emphasis note for research engineering at frontier labs:** P3 + P4 + P6 are the load-bearing core (you build and scale the pipeline); P5 differentiates Anthropic-bound candidates; **P0/P7 are the quiet differentiator** — almost everyone can fine-tune a model; very few can tell you, with a valid confidence interval, whether it actually got better.

---

## Part C — The papers (reading spine)

### C0 · Statistical & probabilistic foundations *(the emphasis — most under-served, highest leverage)*

**Scaling as empirical statistics**
- Kaplan et al., *Scaling Laws for Neural LMs* ([2001.08361](https://arxiv.org/abs/2001.08361))
- Hoffmann et al., *Chinchilla* ([2203.15556](https://arxiv.org/abs/2203.15556))
- Wei et al., *Emergent Abilities* ([2206.07682](https://arxiv.org/abs/2206.07682)) **vs.** Schaeffer et al., *Mirage* ([2304.15004](https://arxiv.org/abs/2304.15004)) — a fundamentally statistical/metric debate

**ICL as inference**
- Xie et al., *ICL as Implicit Bayesian Inference* ([2111.02080](https://arxiv.org/abs/2111.02080))
- Müller et al., *Transformers Can Do Bayesian Inference (PFNs)* ([2112.10510](https://arxiv.org/abs/2112.10510))

**Calibration & uncertainty**
- Guo et al., *On Calibration of Modern Neural Networks* ([1706.04599](https://arxiv.org/abs/1706.04599))
- Kadavath et al., *Language Models (Mostly) Know What They Know* ([2207.05221](https://arxiv.org/abs/2207.05221))
- *Taming Overconfidence in LLMs* ([2410.09724](https://arxiv.org/abs/2410.09724))
- Lakshminarayanan et al., *Deep Ensembles*

**Statistics of evaluation**
- Miller, *Adding Error Bars to Evals* ([2411.00640](https://arxiv.org/abs/2411.00640))
- Bowyer, Aitchison & Ivanova, *Don't Use the CLT* ([2503.01747](https://arxiv.org/abs/2503.01747))
- Card et al., *With Little Power Comes Great Responsibility* ([2010.06595](https://arxiv.org/abs/2010.06595))
- Chiang et al., *Chatbot Arena* ([2403.04132](https://arxiv.org/abs/2403.04132))
- *Dropping Preferences Can Change LLM Rankings* ([2508.11847](https://arxiv.org/abs/2508.11847))

**Prediction-powered inference**
- Angelopoulos et al., *PPI* (Science 2023, [2301.09633](https://arxiv.org/abs/2301.09633)); *PPI++*
- *PPI × E-values* ([2502.04294](https://arxiv.org/abs/2502.04294)); *PPI × conformal/anytime-valid* ([2510.16166](https://arxiv.org/abs/2510.16166))

**Conformal prediction**
- Angelopoulos & Bates, *A Gentle Introduction* ([2107.07511](https://arxiv.org/abs/2107.07511))
- Tibshirani, Barber, Candès, Ramdas, *Conformal Under Covariate Shift*
- Quach et al., *Conformal Language Modeling* (ICLR 2024)

**Anytime-valid inference**
- Vovk & Wang, *E-values*; Howard, Ramdas et al., *time-uniform confidence sequences*

**Causal inference at scale**
- Chernozhukov et al., *Double/Debiased ML*
- Wager & Athey, *Causal Forests / GRF*
- Schölkopf et al., *Toward Causal Representation Learning* ([2102.11107](https://arxiv.org/abs/2102.11107))

**Generalization theory**
- Belkin et al. / Nakkiran et al., *Double Descent*
- Power et al., *Grokking* ([2201.02177](https://arxiv.org/abs/2201.02177))
- Bordelon & Pehlevan, *Feature Learning & Neural Scaling* ([2409.17858](https://arxiv.org/abs/2409.17858))

### C1 · Architectures & scaling
Vaswani et al., *Attention Is All You Need* ([1706.03762](https://arxiv.org/abs/1706.03762)) · Brown et al., *GPT-3* ([2005.14165](https://arxiv.org/abs/2005.14165)) · Gu & Dao, *Mamba* ([2312.00752](https://arxiv.org/abs/2312.00752)) · DeepSeek-AI, *DeepSeek-V3* ([2412.19437](https://arxiv.org/abs/2412.19437)) · Yang et al., *μP / Tensor Programs V*

### C2 · Pretraining & data
Dubey et al., *The Llama 3 Herd* ([2407.21783](https://arxiv.org/abs/2407.21783)) · Abdin et al., *Phi-4* ([2412.08905](https://arxiv.org/abs/2412.08905)) · *FineWeb* · Xie et al., *DoReMi*

### C3 · Post-training & RL
Ouyang et al., *InstructGPT/RLHF* ([2203.02155](https://arxiv.org/abs/2203.02155)) · Rafailov et al., *DPO* ([2305.18290](https://arxiv.org/abs/2305.18290)) · Bai et al., *Constitutional AI* ([2212.08073](https://arxiv.org/abs/2212.08073)) · Lambert et al., *Tülu 3 / RLVR* ([2411.15124](https://arxiv.org/abs/2411.15124)) · Shao et al., *GRPO* ([2402.03300](https://arxiv.org/abs/2402.03300)) · DeepSeek-AI, *R1* ([2501.12948](https://arxiv.org/abs/2501.12948)) · Schulman et al., *PPO* ([1707.06347](https://arxiv.org/abs/1707.06347))

### C4 · Reasoning
Wei et al., *Chain-of-Thought* ([2201.11903](https://arxiv.org/abs/2201.11903)) · Lightman et al., *Let's Verify Step by Step* ([2305.20050](https://arxiv.org/abs/2305.20050)) · Zelikman et al., *STaR*

### C5 · Interpretability, alignment & safety
Elhage et al., *Toy Models of Superposition* · Olsson et al., *Induction Heads* · Anthropic, *Scaling Monosemanticity* (2024) · Meng et al., *ROME / causal tracing* ([2202.05262](https://arxiv.org/abs/2202.05262)) · Frontier system cards (Claude Opus 4.5, GPT-5)

### C6 · Systems & infra
Dao et al., *FlashAttention* ([2205.14135](https://arxiv.org/abs/2205.14135)); FA-2 ([2307.08691](https://arxiv.org/abs/2307.08691)) · Shoeybi et al., *Megatron-LM* ([1909.08053](https://arxiv.org/abs/1909.08053)) · Rajbhandari et al., *ZeRO* ([1910.02054](https://arxiv.org/abs/1910.02054)) · Kwon et al., *vLLM / PagedAttention* ([2309.06180](https://arxiv.org/abs/2309.06180)) · Micikevicius et al., *Mixed-Precision Training*

### C7 · Generative modeling
Ho et al., *DDPM* ([2006.11239](https://arxiv.org/abs/2006.11239)) · Song et al., *Score-Based SDEs* ([2011.13456](https://arxiv.org/abs/2011.13456)) · Lipman et al., *Flow Matching* ([2210.02747](https://arxiv.org/abs/2210.02747)) · Tutorial ([2506.02070](https://arxiv.org/abs/2506.02070)) · Song et al., *Consistency Models* ([2303.01469](https://arxiv.org/abs/2303.01469))

### C8 · Data science frontier
Hollmann et al., *TabPFN* (Nature) · *TabPFN-2.5* ([2511.08667](https://arxiv.org/abs/2511.08667)) · *Chronos / TimesFM / Moirai* (time-series FMs) · (causal + conformal + PPI → C0)

### C9 · AI for science
Abramson et al., *AlphaFold 3* (Nature 2024) · Hayes et al., *ESM3* (Science) · *AlphaProof* (Nature) · Zeni et al., *MatterGen* ([2312.03687](https://arxiv.org/abs/2312.03687)) · GNoME · Herde et al., *Poseidon*

---

## Part D — How to run it

1. **Establish P0 to bedrock first-pass** — read C0's foundations + calibration/eval papers. ~2 weeks. This is the layer everyone skips and labs quietly test; doing it first makes everything downstream legible.
2. **Run the 100-Day Curriculum** as the backbone — data → architecture → systems → RL → interp → theory → science → capstone.
3. **Execute the 30-Build Roadmap in parallel** — the builds are how the papers become muscle memory. *First commit before first paper.*
4. **Use Docs 1–4 as reference** when a phase needs the wider field context.
5. **Loop:** survey maps the edge → build plants you on it → gaps drag you to fundamentals → return. Ship each cycle.

**Frontier filter for 2026+:** weight your time toward (a) reasoning RL & verifiable rewards, (b) interpretability at scale, (c) systems/efficiency (quality-per-dollar frontier), (d) the eval/PPI/conformal statistics layer, and (e) one science or causal thread that's *yours*.

---

## Part E — What research engineering at a frontier lab actually rewards

- **You can build and scale the pipeline, not just use it.** P3 + P6. The 30-build capstone is the single most legible proof.
- **You are empirically rigorous.** P0 + P7. You don't bold the bigger number; you attach a valid interval, run a paired test, and know when the CLT lies.
- **You can read, reproduce, and extend a paper fast.** The whole library trains this; the builds prove it.
- **You have a research thread that's actually yours.** Your CRL × continual-learning, world-models × MARL, and interpretability bets are real frontiers.
- **For Anthropic specifically:** fluency in P5 — interpretability, alignment, eval/CoT monitorability — is the differentiator.

The honest throughline: you don't earn a seat at the frontier by finishing the fundamentals first. You start at the frontier, build badly, let the gaps conscript the immortal fundamentals, and repeat.
