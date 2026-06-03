# Masters in FAIRE — Program Handbook
## Frontiers in AI & Research Engineering · pracha.me/frontier/faire

*A rigorous, self-directed post-master's program for the frontier of AI, calibrated course-by-course to the Stanford / CMU / Georgia Tech flagships and engineered to sit on top of an MS in Data Science + ~7.5 years of ML/AI industry experience. It assumes those as prerequisites and goes where they stop: building, scaling, post-training, interpreting, and statistically validating frontier models. **5 core subjects + qualifying milestones + a thesis.** You don't pass by reading; you pass by shipping reproducible work to a defensible standard.*

**FAIRE** = **F**rontiers in **AI** & **R**esearch **E**ngineering.

**Canvas:** [nthExperiment](README.md) — this repo is the workspace.
**Timeline:** June 2026 → January/February 2027
**Research philosophy:** [How to Do Research](research-philosophy.md) — the craft this program is built on

---

## Program at a glance

| Component | Code | Credits | What it is |
|---|---|---|---|
| Qualifying milestones | FAIRE 500 | gate (0) | Paper reproduction under pressure + statistical defense |
| Foundations: Statistics, Theory & Science of Learning | FAIRE 501 | 6 | The rigor layer |
| Deep Learning & Generative Modeling | FAIRE 502 | 6 | Architectures + diffusion/flow + representations |
| Frontier Language Models | FAIRE 503 | 6 | The pretrain→serve pipeline |
| RL, Post-Training & Alignment | FAIRE 504 | 6 | RLHF→RLVR→reasoning + interpretability/safety |
| Systems & Infrastructure Engineering | FAIRE 505 | 6 | Kernels, distributed training, serving |
| Capstone Thesis | FAIRE 599 | 6 | nanoLM end-to-end + statistical report + defense |
| **Total** | | **36** | Comparable to a top-tier MS course load |

**Each subject = 2–3 arcs (learn-by-building units) + 2 capstones (portfolio-grade, defended).**
**Duration:** 3 terms ≈ 9 months, mapped to the [30-Build Roadmap](seminars/30-builds-roadmap.md).

---

## Positioning & admissions

**Prerequisites (assumed satisfied):**
- *Math & stats maturity* — linear algebra, probability, statistical inference (MS DS + TA in Causal Inference, Risk Analytics, Statistical/Advanced Data Analysis)
- *ML/DL foundations* — supervised learning, neural nets, optimization (MS DS: ML, Advanced Deep Learning; 6.5+ yrs industry ML)
- *Software & systems fluency* — Python, PyTorch/JAX, basic distributed systems
- *Research literacy* — reading and reproducing papers (COLM/CVPR-workshop work)

**What FAIRE adds beyond an MS DS:** an MS DS makes you a *consumer* of ML methods. FAIRE makes you a *producer at the frontier* — implement the transformer/MoE/attention stack from scratch, run RLVR/GRPO, train multi-GPU, write CUDA/Triton kernels, do mechanistic interpretability, and attach **valid statistical guarantees** to every claim.

**Benchmark of rigor:** each subject is at or beyond the equivalent flagship graduate course (CMU 10-414, Stanford CS336, CS234, CMU 36-705), assessed by reproducible build + defended write-up.

---

## Grading & assessment rubric

Every arc and capstone graded on five axes (capstone passes at ≥4 on Correctness and Communication, ≥3 on rest):

| Axis | What it measures |
|---|---|
| **Correctness & reproducibility** | Does it run from a clean clone and produce the claimed result? |
| **Statistical rigor** | Are claims backed by valid uncertainty (CIs, paired tests, calibration)? *(the FAIRE signature)* |
| **Depth / from-scratch-ness** | Implemented from first principles vs. wired together? |
| **Engineering quality** | Efficiency, profiling, clean code, tests |
| **Communication** | A write-up a lab reviewer could read and trust |

**Honor code:** every deliverable ships publicly (repo + write-up). *First commit before first paper.* No private half-builds count.

---

## FAIRE 500 — Qualifying Milestones *(the gate)*

Two defended milestones before thesis work:

- **Q1 — Reproduce a paper cold.** Pick a frontier method paper; reproduce its core result from scratch in a bounded time (1 week), with ablations and a write-up of where you diverged. *Proves: research literacy + implementation speed.*
- **Q2 — Statistical defense.** Take any model comparison and defend it statistically: valid intervals, paired test, power analysis, calibration check — and identify one way the naive analysis misleads. *Proves: the FAIRE-signature rigor.*

---

## FAIRE 501 — Foundations: Statistics, Theory & Science of Learning
**6 credits · Term 1**

*Peer equivalence: CMU 36-705 + 10-725 + Stanford STATS 305/EE364A, extended with AI-era statistics.*

**Outcomes:**
1. Derive training as MLE; reason about generalization in the overparameterized regime
2. Produce valid uncertainty for any ML metric (bootstrap, paired tests, exact vs. CLT); plan experiment power
3. Implement prediction-powered inference (PPI) and conformal prediction with guarantees
4. Diagnose and fix model calibration; choose proper scoring rules
5. Reproduce a science-of-learning phenomenon (grokking / feature-learning) and explain the mechanism

**Arcs:** I.A Inference toolkit from scratch · I.B PPI + conformal on an LLM · I.C grokking/feature-learning reproduction

**Capstones:**
- **501.1 "An Honest Eval"** — defensible statistical evaluation of two models (paired tests, bootstrap vs. exact, PPI, calibration)
- **501.2 "Math + Stats for Frontier AI"** — from-scratch notebook-text (likelihood, fitted scaling laws, calibration, conformal)

**Required reading:** Miller (Adding Error Bars to Evals); Bowyer et al.; Angelopoulos PPI; Angelopoulos–Bates conformal; Guo calibration; Kadavath; Chiang Chatbot Arena; Power grokking

---

## FAIRE 502 — Deep Learning & Generative Modeling
**6 credits · Term 1–2**

*Peer equivalence: Stanford CS230/CS231N, CMU 11-785/10-423, GTech CS7643.*

**Outcomes:**
1. Implement transformer block + attention family (MHA/MQA/GQA/MLA) from scratch; analyze KV-cache tradeoff
2. Implement SSM/Mamba block; reason about sub-quadratic sequence modeling
3. Derive and implement diffusion and flow matching; unify under generator-matching
4. Build a JEPA-style representation learner and probe its features

**Arcs:** II.A transformer + attention zoo · II.B Mamba block vs. attention · II.C diffusion→flow matching from scratch

**Capstones:**
- **502.1** Flow-matching generative model on a real modality (vs.-diffusion write-up)
- **502.2** I-JEPA-lite representation learner with a probing study *(JEPA+MPC research thread)*

**Required reading:** Vaswani; Gu–Dao Mamba; Ho DDPM; Lipman Flow Matching; Song Consistency; He MAE; Assran I-JEPA

---

## FAIRE 503 — Frontier Language Models
**6 credits · Term 2**

*Peer equivalence: Stanford CS336 (Language Modeling from Scratch) + CS224N, CMU 11-711.*

**Outcomes:**
1. Build a data pipeline (curate, dedup, mix) and train a tokenizer; reason about data-centric scaling
2. Pretrain a decoder-only LM (RoPE+GQA) and a MoE variant with auxiliary-loss-free balancing
3. Stand up inference (paged-KV, speculative decoding, quantization)
4. Read a frontier technical report and reproduce one recipe component with ablations

**Arcs:** III.A data & tokenizer · III.B nanoLM base + MoE · III.C inference & serving

**Capstones:**
- **503.1 nanoLM pretraining** — small model end-to-end + frontier-style tech report *(thesis spine begins)*
- **503.2** Reproduce a frontier component (MLA / MTP / aux-loss-free MoE) with ablations

**Required reading:** Llama 3 Herd; DeepSeek-V3; Chinchilla; Phi-4; Tülu 3

---

## FAIRE 504 — RL, Post-Training & Alignment
**6 credits · Term 2–3 · Honors core**

*Peer equivalence: Stanford CS234, CMU 10-403 Deep RL + alignment/interpretability frontier.*

**Outcomes:**
1. Implement SFT, reward modeling, PPO-RLHF, and DPO (+ variants); explain reward over-optimization
2. Implement RLVR/GRPO on a verifiable task; reproduce reasoning-length-grows curve
3. Train and analyze a sparse autoencoder; locate circuits (induction heads) via activation patching
4. Run a Constitutional-AI/RLAIF loop; author a mini system card with reward-hacking + CoT analysis

**Arcs:** IV.A SFT+DPO · IV.B GRPO/RLVR from scratch · IV.C SAE + circuit analysis

**Capstones:**
- **504.1 "Mini-R1"** — nanoLM-base → SFT cold-start → GRPO → distill, with report
- **504.2** Alignment & safety study — RLAIF loop + reward-hacking probe + mini system card

**Required reading:** InstructGPT; DPO; GRPO/DeepSeekMath; R1; Constitutional AI; Toy Models of Superposition; Induction Heads; Scaling Monosemanticity

---

## FAIRE 505 — Systems & Infrastructure Engineering
**6 credits · Term 3 · Honors core**

*Peer equivalence: CMU 10-414/714 Deep Learning Systems + 15-418 Parallel Programming + 11-868 LLM Systems.*

**Outcomes:**
1. Write a fused Triton kernel + mini IO-aware (FlashAttention-style) attention; reason from roofline
2. Train multi-GPU with FSDP/ZeRO + tensor/pipeline parallelism; diagnose training instability
3. Build an efficient serving stack (paged-KV, continuous batching, FP8/INT4); benchmark throughput/latency/cost

**Arcs:** V.A Triton kernel + IO-aware attention · V.B distributed training (FSDP) · V.C efficient serving

**Capstones:**
- **505.1** Multi-GPU nanoLM training stack (FSDP + custom kernel + stability log + MFU report)
- **505.2** Mini-vLLM serving system with full throughput/latency/cost study

**Required reading:** FlashAttention 1/2/3; Megatron-LM; ZeRO; vLLM/PagedAttention; mixed-precision; DeepSeek-V3 hardware paper

---

## FAIRE 599 — Capstone Thesis
**6 credits · Term 3 (integrative)**

**The thesis spine, composed.** Across 503–505 you build nanoLM, make it reason and reverse-engineer it (504), and scale + serve it (505). The thesis integrates everything:

> **nanoLM, end to end** — data → pretrain → post-train (SFT→RLVR) → interpret (SAE) → serve — **plus a 501-grade statistical validation report** establishing, with valid uncertainty, that each stage actually improved the model.

**Defense:** written report + recorded walkthrough justifying every design and statistical claim.

**Honors option:** extend toward one research bet — CRL × continual learning, world-models × MARL, or mechanistic interpretability — as a small original contribution.

---

## Academic calendar

| Term | Weeks | Subjects | Milestone |
|---|---|---|---|
| **Term 1** | 1–12 | 501 + start 502 | Q1 + Q2 quals (FAIRE 500) |
| **Term 2** | 13–24 | finish 502 · 503 · start 504 | 503.1 nanoLM (thesis spine begins) |
| **Term 3** | 25–36 | finish 504 · 505 · **599 thesis** | Thesis defense |

*Maps onto the [30-Build Roadmap](seminars/30-builds-roadmap.md) as the literal calendar — weekly bricks are arcs, monthly arcs are capstones.*

---

## What the degree certifies

On completion, with public reproducible evidence, you can claim:

- **Build & scale a frontier model end to end** (503 + 505 + thesis)
- **Do reasoning RL, post-training, and interpretability** (504) — the 2024–2026 frontier and the Anthropic-defining skill set
- **Validate any claim statistically** (501 + quals) — the rare rigor that reads as seniority
- **Implement, derive, and reproduce from first principles** (all subjects + Q1) — research literacy, demonstrated

---

## Curriculum map (what lives where in this repo)

| FAIRE component | nthExperiment resource |
|-----------------|----------------------|
| Seminar reading | [seminars/frontier-ai/](seminars/frontier-ai/) — 16 topics, 327 papers |
| Field maps | [seminars/](seminars/) — ML, DL, DS, Applied Stats surveys |
| Library / concepts | [library/](library/) — foundations, deep learning, RL, causal, systems |
| Reads & references | [library/reads-and-references/](library/reads-and-references/) |
| Lecture myself | [lecture-myself/](lecture-myself/) — 10 tracks, university-level notes |
| Experiments / builds | [experiments/](experiments/) + [tiny-experiments/](tiny-experiments/) |
| 30-build roadmap | [seminars/30-builds-roadmap.md](seminars/30-builds-roadmap.md) |
| 100-day curriculum | [seminars/frontier-models-100day-curriculum.md](seminars/frontier-models-100day-curriculum.md) |
| Weekly reviews | [reviews/](reviews/) |
| Now (what's active) | [now.md](now.md) |

---

*FAIRE is the nthExperiment library, organized as a degree you grant yourself — and earn — by shipping.*
