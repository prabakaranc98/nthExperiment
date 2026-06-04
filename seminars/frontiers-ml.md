# Frontiers in Machine Learning — A Survey & Map

*Doc 1 of 4. This one covers **machine learning as a field** — the theory-of-learning and learning-paradigm layer that is broader than, and contains, deep learning. Where a topic is specifically about neural networks it lives in Frontiers in Deep Learning; where it's about extracting reliable knowledge/decisions from data it lives in Frontiers in Data Science.*

---

## What "frontier" means in ML

ML's native frontier is the **question of learning itself**: what can be learned, from how much data, by which procedure, with what guarantees, and why does it generalize? The capability frontier (what systems do) is downstream; ML's frontier is the *theory and paradigms of learning*.

**Field-maturity tell:** when a paradigm earns a stable textbook chapter, it has left the frontier and become a fundamental.

Concept definitions for everything below live in the [concept library](../library/bricks/README.md).

---

## 1. Learning paradigms — the shift to general learners

The deepest current shift is **away from task-specific supervised learning toward pretrained general learners** that adapt at inference.

| Paradigm | One-line frontier | Status (2026) |
|---|---|---|
| **Self-supervised learning** | What objective yields the most transferable representations? | Engine behind every foundation model; objective design still open. |
| **In-context learning (ICL)** | ICL as *amortized Bayesian inference* — train once over a task prior, one forward pass approximates the posterior. | Now realized literally beyond toy settings (see PFNs below). |
| **Meta-learning** | "Learning to learn" / few-shot adaptation theory. | Largely subsumed by ICL, still the right lens for adaptation guarantees. |
| **Continual / lifelong learning** | A stream of tasks without catastrophic forgetting. | Genuinely open; no method matches biological retention. |

### ICL has gone operational: Prior-data Fitted Networks (PFNs)

PFNs make ICL-as-Bayesian-inference concrete: pretrain a transformer on synthetic datasets drawn from a prior, then return a posterior predictive in a single forward pass at test time. The 2025-2026 frontier is no longer the toy demonstration but **deployed amortized inference**:

- **Tabular foundation models** — TabPFN-2.5 ([2511.08667](https://arxiv.org/abs/2511.08667)) scales the recipe to ~50k rows / 2k features and reports a 100% win rate vs. default XGBoost on small-to-medium tables. TabICL v2 (SODA/Inria) competes on speed and large tables. Tabular ML now has a credible foundation-model default.
- **Amortized causal & survival inference** — CausalPFN (causal-effect estimation) and SurvivalPFN port the prior-fitting recipe to causal and time-to-event tasks.

**Known limit:** PFNs assume a *fixed* prior; misspecified priors (e.g. non-identifiable causal settings) give asymptotically inconsistent posteriors. Choosing and adapting the prior is the live problem.

### Continual learning: the surprising RL result

The 2025-2026 reframing centers on a finding that **RL fine-tuning is far more forgetting-robust than SFT**, often without explicit continual-learning machinery — shifting attention from regularization tricks toward training *objective* as the lever. In production, "continual learning" increasingly means external/agentic memory layers and test-time learning rather than weight updates; benchmarks like MemoryAgentBench (ICLR 2026) now score retrieval, test-time learning, long-range use, and *selective forgetting* as distinct competencies.

> **Open problem:** A unified account of when and why pretraining + adaptation beats task-specific training — and what the right pretraining prior is.

---

## 2. Statistical & learning theory — the theory of generalization

Classical theory (VC dimension, uniform convergence, bias-variance) explains small models and *fails* for overparameterized, interpolating models. The frontier is a new theory of generalization.

| Thread | The open question |
|---|---|
| **Benign overfitting & double descent** | Why models that perfectly fit even noisy data still generalize. |
| **Implicit bias of optimization** | SGD/Adam/Muon don't just minimize loss — they prefer particular solutions. Characterizing the preference is central. |
| **PAC-Bayes & information-theoretic bounds** | The most promising route to *non-vacuous* bounds for modern models. |
| **ICL theory** | Formalizing ICL as gradient descent / Bayesian inference, not analogy. |
| **Grokking** | Delayed generalization as signal migrating from a "reservoir" into a signal channel as the kernel evolves. |

**2025-2026 movement:** unifying accounts now try to explain benign overfitting, double descent, implicit bias, and grokking under one frame — analyzing generalization in the network's *output space* and linking feature emergence to learning dynamics. These also begin to explain *why* optimizers like Muon help, tying theory back to practice.

(The deep-net-specific version — feature-learning vs. NTK, scaling-law theory — is in the DL doc.)

> **Open problem:** A *predictive* theory that, given (data, architecture, optimizer), forecasts generalization — not just explains it after the fact.

---

## 3. Reinforcement learning — the learning-from-interaction frontier

| Thread | What's frontier (2026) |
|---|---|
| **Reward design** | Scalable, hard-to-hack reward. RL with verifiable rewards (RLVR) — exact-answer checks, unit tests, proof checkers — is the current answer. |
| **Reasoning RL** | Critic-free policy optimization (GRPO, DeepSeek-R1) inducing reasoning. Successors now diverge: **DAPO** for training stability, **GSPO** (sequence-level) for correctness; refinements like Scaf-GRPO and median-centered baselines target plateaus and small-rollout instability. |
| **Open-ended RLVR** | Extending verifiable reward beyond math/code via reference-based reward chains and multiple-choice reformulation. |
| **Offline RL** | Good policies purely from logged data. |
| **Model-based RL & world models** | Learn a predictive model and plan in it (Dreamer lineage); now converging with *causal* world models. |
| **Multi-agent RL** | Coordination, emergent communication, zero-shot coordination. |

### The central RL debate: elicit vs. create

Does RL produce *new* capability or merely surface what pretraining already learned? This is the field's sharpest live disagreement.

- **Elicitation camp** — "Does RL Really Incentivize Reasoning Beyond the Base Model?" ([2504.13837](https://arxiv.org/abs/2504.13837)): RLVR models win at low *k* but base models match or exceed them at high pass@*k*; RL-found reasoning paths already sit in the base sampling distribution.
- **Creation camp** — ProRL ([2505.24864](https://arxiv.org/abs/2505.24864)): with *prolonged* RL plus KL control, reference resets, and diverse tasks, models reach solutions the base model never produces under heavy sampling.

The disagreement turns on training duration and evaluation methodology; resolving it is prerequisite to knowing how far RLVR can scale.

> **Open problem:** Sample-efficient, gaming-resistant RL — and a theory of when RL *creates* capability vs. merely *elicits* it.

---

## 4. Causal machine learning — from correlation to mechanism

| Thread | Frontier |
|---|---|
| **Causal discovery** | Recovering causal graphs from observational + interventional data. |
| **Invariance & OOD generalization** | IRM and successors: representations stable across environments because they capture mechanisms, not spurious correlations. |
| **Causal representation learning (CRL)** | Recovering latent causal variables and mechanisms from raw data (the Schölkopf program). → [2102.11107](https://arxiv.org/abs/2102.11107) |
| **Causality × foundation models** | Do large pretrained models implicitly encode causal structure, and can we extract/steer it? |

**2025-2026 movement:**

- **Identifiability with fewer environments / finite samples** — new results give finite-sample recovery guarantees from a *sublinear* (e.g. logarithmic) number of multi-node interventions, pushing CRL past the old "infinite environments" idealization.
- **Causal foundation models** — amortized PFN-style priors for causal inference (CausalFM / CausalPFN line) and causal *world* models that connect representation to decision-making.
- **Caveat:** priors not restricted to identifiable settings are provably misspecified, so foundation-scale causal inference inherits the identifiability question rather than escaping it.

> **Open problem:** Identifiability — under what assumptions can latent causal structure be recovered, and can foundation-scale models do it *without* those assumptions?

---

## 5. Probabilistic & Bayesian ML — learning under uncertainty

- **Amortized & variational inference** — networks that output posteriors; the bridge from Bayesian inference to deep learning and the conceptual root of PFNs/ICL. The 2025-2026 thread asks whether transformers learn posterior inference *directly from the prompt*.
- **Simulation-based inference (SBI)** — Bayesian inference when you can simulate but not write the likelihood; central in the sciences.
- **Calibration & uncertainty quantification** — conformal prediction and friends; full treatment in the DS doc.

> **Open problem:** Scalable, well-calibrated uncertainty for foundation-scale models.

---

## 6. Optimization — mostly solved, with a live edge

The big 2025-2026 story is that **Muon broke Adam's monopoly at scale.**

- **Muon and its family** — orthogonalized/spectral-norm momentum updates report ~2× compute efficiency vs. AdamW ([2502.16982](https://arxiv.org/abs/2502.16982)) and are now deployed in trillion-parameter training (MuonClip in Kimi-scale runs, orthogonalization in GLM-4.5-scale models). Variants: AdaMuon, NorMuon.
- **Hyperparameter transfer** — μP / Tensor Programs: tune small, transfer to large.
- **Second-order methods** — Shampoo / SOAP remain the practical competition.
- **Optimization-as-theory** — implicit-bias and edge-of-stability questions; generalization theory now starts to explain *why* Muon-style updates help.

> **Open problem:** An optimizer provably better than Adam at scale — and a theory of why the working ones work.

---

## 7. Cross-cutting: AutoML, efficiency, constraints

- **AutoML / NAS** — folding into "let a model design the model"; LLM-driven pipeline and architecture search.
- **Learning under constraints** — fairness, privacy (DP-SGD), robustness as first-class objectives.
- **Efficient learning** — active learning, curriculum, data valuation; and amortized inference (PFNs) as a route to near-zero-cost fitting on small data.

---

## How to stay on the ML frontier

Stand where a *paradigm* is shifting. Right now:

- General learners (and amortized inference) replacing task-specific training.
- RL maturing into a capability engine — with the elicit-vs-create question still open.
- Causality fusing with representation learning and foundation models.
- Muon-class optimizers displacing Adam at scale.

Track NeurIPS / ICML / ICLR workshops, follow the ~30 people per thread, and watch for the "first survey" tell.

**The immortal ML fundamentals:** probability & information theory; statistical learning theory; optimization (convex + non-convex); the core of RL (MDPs, DP, policy gradients); and the foundations of causal inference (SCMs, do-calculus, identification). Everything frontier above is a recombination of these.

*Next: Frontiers in Deep Learning (architectures, diffusion/flow, DL theory & feature learning, mechanistic interpretability, neural operators).*
