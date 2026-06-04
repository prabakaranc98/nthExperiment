# Applied Statistics for the Modern AI Era — A Survey & Map

*Doc 4 of 4. The inferential machinery underneath the three frontier docs — grounded in the actual papers and on-the-ground practice used in frontier AI today. In the foundation-model era the job of statistics moved — from building the model to the **evaluation, decision, and trust layer**: is model B actually better than A, or is that a noise-sized delta? How confident is this output, with a guarantee? What happens if I intervene?*

> **Read this as a map, not a textbook.** Each section names the core idea, the canonical source, and the on-the-ground use. Concept stubs live in the [concept library](../library/bricks/README.md).

---

## 1. Estimation & inference, reframed

| Distinction | What it means | Why it bites in AI |
|---|---|---|
| **Likelihood view of training** | Fitting *is* maximum likelihood — cross-entropy is negative log-likelihood; weight decay is a Gaussian MAP prior | The bridge that makes all of ML a statistics problem |
| **Estimate vs. inference** | A number (53.2% accuracy) vs. how sure you are of it | The literature is flooded with estimates, starved of inference |
| **Confidence vs. credible interval** | Frequentist coverage vs. Bayesian posterior probability | Know which one you are claiming before you bold it |

---

## 2. Resampling & uncertainty — the workhorses

- **Bootstrap** (Efron) — resample with replacement for error bars on any statistic, including non-standard ones (AUC, F1, eval accuracy, Elo). The method behind LMArena's 95% CIs.
- **Cross-validation done right** — nested CV, group/temporal splits to prevent leakage.
- **Permutation tests** — assumption-light "is this difference real?"
- **Power analysis for ML** — Card et al., *With Little Power Comes Great Responsibility* ([2010.06595](https://arxiv.org/abs/2010.06595)): many NLP experiments are underpowered. Planning sample size before an eval is still rare and still valuable.

---

## 3. Bayesian & hierarchical modeling

- **Bayesian workflow** (Gelman et al.) — prior → likelihood → posterior, with prior/posterior predictive checks.
- **Hierarchical / multilevel models** — partial pooling across groups. Applied directly to evals: **HiBayES** models per-item and per-task structure instead of collapsing an eval to one flat average — the right default when benchmarks are small and heterogeneous (see §7).
- **Amortized-inference bridge** — PFNs reframe "do Bayesian inference" as "pretrain on a task prior, predict the posterior in one forward pass." Hierarchical Bayes is its conceptual ancestor.

---

## 4. Experimental design & causal identification

| Tool | Source | One-line use |
|---|---|---|
| **Potential outcomes & identification** | Neyman–Rubin | What a causal effect is + the assumptions (ignorability, overlap, SUTVA) that identify it |
| **A/B testing at scale** | — | Power/MDE, variance reduction (CUPED), peeking, network/novelty effects |
| **Double / debiased ML** | Chernozhukov et al. | ML nuisance estimators while keeping valid inference on the effect |
| **Causal forests / GRF** | Wager–Athey | Nonparametric heterogeneous treatment effects |
| **Bandits** | — | Learn while deciding; the bridge to RL |

**On the ground:** shipping an AI feature *is* an intervention. "The model improved the metric" is a confounded observational claim unless the rollout was randomized — increasingly relevant as agentic products ship continuous online changes.

---

## 5. Multiple testing & the reproducibility crisis in AI

- **FWER vs. FDR** — Bonferroni vs. Benjamini–Hochberg when you run many comparisons.
- **The garden of forking paths** — "we tried configs until one beat SOTA" is multiple testing in disguise.
- **Held-out discipline & decontamination** — frontier reports (Claude, Llama, and others) document fuzzy decontamination (n-gram / long-window overlap filtering) precisely because contamination invalidates the measurement.
- **Benchmark saturation & leakage of public sets** — as static benchmarks saturate, labs lean on **private / held-out and dynamically refreshed** test sets; the open debate is whether any public leaderboard score can stay uncontaminated for long.

---

## 6. Calibration & proper scoring — trustworthy probabilities

**RLHF induces overconfidence.** Base models can be reasonably calibrated (Kadavath et al., *Language Models (Mostly) Know What They Know*, 2022), but RLHF sharpens output distributions and degrades calibration; reward models show a *systematic bias toward confident-sounding answers regardless of correctness* (*Taming Overconfidence in LLMs*, [2410.09724](https://arxiv.org/abs/2410.09724)).

**The reasoning-model wrinkle (2025–2026).** Long-chain reasoners (o-series, DeepSeek-R1 and successors) shifted the picture, and the evidence cuts both ways:

- Verbalized confidence on reasoners is often **>85% even on wrong answers**, and over-reasoning can *worsen* calibration (*Don't Think Twice! Over-Reasoning Impairs Confidence Calibration*, [2508.15050](https://arxiv.org/abs/2508.15050)).
- Yet RL-trained reasoners can be **better calibrated than their instruction-tuned base**, and their confidence sometimes drops on longer, failing chains (*Thinking Out Loud: Do Reasoning Models Know When They're Right?*, [2504.06564](https://arxiv.org/abs/2504.06564)).

**Fixes & honest scoring:**

- Recalibration — temperature scaling, Platt / isotonic.
- Proper scoring rules (log loss, Brier) — reward honest probabilities; they cannot be gamed by misreporting confidence.
- Self-consistency / semantic-entropy signals as cheap confidence proxies for free-form generation.

---

## 7. The statistics of LLM evaluation — the most AI-era-specific frontier

The field's actual bottleneck. Evals are experiments; treat them like one.

| Question | Method | Source |
|---|---|---|
| Error bars on a single eval? | Treat items as a sample from a super-population; CLT/paired formulas | Miller, *Adding Error Bars to Evals* (Anthropic, [2411.00640](https://arxiv.org/abs/2411.00640)) |
| Is A better than B? | **Paired** test on shared questions (kills cross-item variance) | Miller, *ibid.* |
| Small / specialized benchmark? | **Don't trust the CLT** — it dramatically understates uncertainty below a few hundred items; use Bayesian or exact frequentist intervals | Bowyer, Aitchison & Ivanova, ICML 2025 position paper ([2503.01747](https://arxiv.org/abs/2503.01747)) |
| Per-item / per-task structure? | Hierarchical Bayes (HiBayES) instead of one flat average | §3 |

**Pairwise / preference evaluation — LMArena (Chatbot Arena).** Chiang et al. ([2403.04132](https://arxiv.org/abs/2403.04132)) fit a **Bradley–Terry** model by MLE with **bootstrap CIs**, **active sampling** of informative matchups, and **E-values** for sequential validity. The board reports CIs because top models routinely sit inside overlapping intervals — rank order is partly statistical noise.

**Open debates as of 2025–2026:**

- **Rankings are fragile.** Dropping a tiny worst-case fraction of votes can flip Bradley–Terry rankings (*Dropping Just a Handful of Preferences Can Change Top LLM Rankings*, [2508.11847](https://arxiv.org/abs/2508.11847)).
- **The leaderboard illusion.** Singh et al. (*The Leaderboard Illusion*, [2504.20879](https://arxiv.org/abs/2504.20879)) argue private multi-variant testing, selective disclosure, and data-access asymmetry let a few labs overfit Arena dynamics (length, formatting); LMArena disputes the framing, noting overlapping CIs imply much of the gap is noise. Style/length **controls** are now standard mitigations — and themselves contested.
- **LLM-as-judge biases.** Position bias (favoring first/second option) and verbosity bias (favoring longer answers). Treat a judge as a measurement instrument with known, correctable biases — not an oracle (see §10 for debiasing the resulting estimate).

---

## 8. Conformal prediction — distribution-free guarantees

| Setting | Method | Source |
|---|---|---|
| Any model, i.i.d.-ish | **Split conformal** — prediction sets with finite-sample coverage ≥ 1−α under exchangeability | Vovk; Angelopoulos & Bates ([2107.07511](https://arxiv.org/abs/2107.07511)) |
| Test ≠ train | Weighted conformal under covariate shift | Tibshirani, Barber, Candès, Ramdas (2019) |
| LLM **generation** | Output sets with correctness guarantees | Quach et al., *Conformal Language Modeling* (ICLR 2024) |
| LLM **evaluation** | Conformal intervals on LLM-as-judge scores; set width as a per-instance judge-reliability signal | *Analyzing Uncertainty of LLM-as-a-Judge* ([2509.18658](https://arxiv.org/abs/2509.18658)) |

---

## 9. Anytime-valid inference & e-values — a live statistics frontier

The classic test breaks if you peek. **Sequential / always-valid** inference fixes it and is built for monitoring and online experimentation.

- **E-values** (Vovk & Wang) and **always-valid p-values / confidence sequences** (Howard, Ramdas et al.; Waudby-Smith & Ramdas) — inference that stays valid no matter when you stop or how often you look.
- **Why it's AI-relevant** — continuous monitoring of deployed models, drift detection, and online A/B experimentation all need anytime-valid guarantees. LMArena already uses E-values for sample-efficient sequential ranking, making this the rare frontier method already in production.

---

## 10. Prediction-powered inference (PPI) — the method built for the AI era

The single most "modern AI era" statistical method: get honest error bars when most of your labels came from a model.

- **The idea** (Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, *Science* 2023; [2301.09633](https://arxiv.org/abs/2301.09633)) — a few expensive gold labels plus a large pile of cheap ML predictions. PPI uses the predictions to shrink variance *while* using the gold labels to debias — **valid CIs and p-values** with the power of large ML-labeled data. **PPI++** tunes it for efficiency.
- **Why it matters** — the principled answer to "I labeled my dataset with an LLM; can I trust the estimate?" Applied to LLM evals, GWAS, computer-vision rankings, and protein-fitness prediction.

**The 2025–2026 PPI wave (autoeval moved from method to default):**

| Variant | What it adds | Source |
|---|---|---|
| **AutoEval Done Right** | PPI for model evaluation with synthetic/AI labels; up to ~50% larger effective human sample on GPT-4 experiments; demos on Chatbot Arena pairwise comparisons | Boyeau, Angelopoulos, Yosef, Malik, Jordan — ICML 2025 ([2403.07008](https://arxiv.org/abs/2403.07008)) |
| **PPI → E-values** | Anytime-valid PPI | [2502.04294](https://arxiv.org/abs/2502.04294) |
| **PPI → conformal / anytime-valid** | Unites §8–§10 | [2510.16166](https://arxiv.org/abs/2510.16166) |
| **Prediction-powered ranking** | Debiased *rankings* from noisy LLM judges with few human labels (PRECISE, AAAI) | Amazon Science — name the source; verify ID before citing |

**On the ground:** PPI is the tool for the now-ubiquitous workflow of *labeling data with a model and wanting honest error bars* — a skill almost no one has and everyone needs.

---

## 11. The pitfalls (statistical thinking as a shield)

The errors that quietly invalidate AI work:

- **Data leakage** — test info in training; the most common eval killer.
- **Benchmark contamination** — public test sets memorized during pretraining.
- **Simpson's paradox** — a trend that reverses once you condition.
- **Selection / survivorship bias** — you only see the runs that finished.
- **Base-rate neglect** — ignoring priors when reading a metric.
- **Statistical vs. practical significance** — a real but useless delta.

Most "surprising" results are one of these in disguise.

---

## How to use applied statistics at the frontier

| Situation | Reach for | Section |
|---|---|---|
| Eval delta between two models | Bootstrap CIs + **paired** test | §7 |
| Small / specialized benchmark | Bayesian or exact intervals (not the CLT) | §7 |
| You labeled data with a model | **PPI / AutoEval** | §10 |
| Guaranteed prediction or eval intervals | **Conformal** | §8 |
| Monitoring or online experiments | **Anytime-valid** tests / e-values | §9 |
| Trustworthy probabilities | Calibration + proper scoring | §6 |
| What an intervention actually does | DML / causal forests + randomization | §4 |

**The immortal statistical core:** probability (distributions, expectation, CLT — *and its limits*); estimation & inference (likelihood, sampling distributions, bootstrap); experimental design & causal inference; regression and the bias-variance decomposition; and Bayesian inference. Sections 7–10 are where this core is *actively extending* into the AI era — precisely the frontier you want to stand on.
