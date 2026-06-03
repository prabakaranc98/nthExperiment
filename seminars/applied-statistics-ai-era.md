# Applied Statistics for the Modern AI Era — A Survey & Map

*Doc 4 of 4. The inferential machinery underneath the three frontier docs — grounded in the actual papers and on-the-ground practice used in frontier AI today. In the foundation-model era the job of statistics moved — from building the model to the **evaluation, decision, and trust layer**: is model B actually better than A, or is that a noise-sized delta? How confident is this output, with a guarantee? What happens if I intervene?*

---

## 1. Estimation & inference, reframed

- **The likelihood view of training.** Fitting a model *is* maximum likelihood — cross-entropy is negative log-likelihood, and weight decay is a Gaussian MAP prior. This is the bridge that makes all of ML a statistics problem.
- **Estimate vs. inference.** A number (53.2% accuracy) vs. how sure you are about it. The AI literature is flooded with estimates and starved of inference.
- **Confidence vs. credible intervals.** Frequentist coverage vs. Bayesian posterior probability. Know which you're claiming.

---

## 2. Resampling & uncertainty — the workhorses

- **Bootstrap** (Efron) — resample with replacement to get error bars on any statistic, including non-standard ones (AUC, F1, eval accuracy, Elo). What Chatbot Arena uses for 95% CIs.
- **Cross-validation done right** — nested CV, group/temporal splits to prevent leakage.
- **Permutation tests** — assumption-light "is this difference real?"
- **Power analysis for ML.** Card et al., *With Little Power Comes Great Responsibility* ([2010.06595](https://arxiv.org/abs/2010.06595)) — many NLP experiments are underpowered. Planning sample size before an eval is rare and valuable.

---

## 3. Bayesian & hierarchical modeling

- **Bayesian workflow** (Gelman et al.) — prior → likelihood → posterior, with prior/posterior predictive checks.
- **Hierarchical / multilevel models** — partial pooling across groups. Applied directly to AI evals: **HiBayES** — a hierarchical Bayesian framework for eval statistics, modeling per-item and per-task structure instead of treating an eval as one flat average.
- **The amortized-inference bridge.** PFNs reframe "do Bayesian inference" as "pretrain on a task prior, predict the posterior in one forward pass" — hierarchical Bayes is its conceptual ancestor.

---

## 4. Experimental design & causal identification

- **Potential outcomes & identification** (Neyman–Rubin) — what a causal effect is and the assumptions (ignorability, overlap, SUTVA) that let data identify it.
- **A/B testing at scale** — power/MDE, variance reduction (CUPED), peeking, network effects, novelty.
- **Double / debiased ML** (Chernozhukov) — use ML nuisance estimators while keeping valid inference on the causal effect.
- **Causal forests / GRF** (Wager–Athey) — nonparametric heterogeneous treatment effects.
- **Bandits** — learn while deciding; the bridge to RL.

**On the ground:** shipping an AI feature *is* an intervention, and "the model improved the metric" is almost always a confounded observational claim unless it was randomized.

---

## 5. Multiple testing & the reproducibility crisis in AI

- **FWER vs. FDR** — Bonferroni vs. Benjamini–Hochberg when you run many comparisons.
- **The garden of forking paths** — "we tried configs until one beat SOTA" is multiple testing in disguise.
- **Held-out discipline & decontamination.** Frontier labs now report fuzzy decontamination procedures (n-gram/20-gram overlap filtering in Claude/Llama reports) precisely because contamination invalidates the measurement.

---

## 6. Calibration & proper scoring — trustworthy probabilities

**RLHF induces overconfidence.** Base models can be reasonably calibrated (Kadavath et al., *Language Models (Mostly) Know What They Know*, 2022), but RLHF sharpens output distributions and degrades calibration; reward models exhibit a *systematic bias toward confident-sounding answers regardless of correctness* (*Taming Overconfidence in LLMs*, [2410.09724](https://arxiv.org/abs/2410.09724)).

- **Fixes** — temperature scaling, Platt/isotonic recalibration.
- **Proper scoring rules** (log loss, Brier) — reward honest probabilities because they can't be gamed by misreporting confidence.

---

## 7. The statistics of LLM evaluation — the most AI-era-specific frontier

The field's actual bottleneck.

- **Evals are experiments.** Miller, *Adding Error Bars to Evals* (Anthropic, [2411.00640](https://arxiv.org/abs/2411.00640)) — treat eval questions as drawn from an unseen super-population; gives formulas for analyzing eval data, comparing two models with a **paired test** (same questions → reduce variance), and planning (how many questions to detect a given gap). The blunt point: industry bolds the SOTA number without testing significance.
- **Don't blindly use the CLT.** Bowyer, Aitchison & Ivanova ([2503.01747](https://arxiv.org/abs/2503.01747)) — CLT-based error bars are fine for thousand-item benchmarks but *dramatically underestimate* uncertainty on small, specialized benchmarks. Use Bayesian or exact frequentist intervals.
- **Pairwise/preference evaluation — Chatbot Arena.** Chiang et al. ([2403.04132](https://arxiv.org/abs/2403.04132)) fit a **Bradley–Terry** model by MLE, with **bootstrap confidence intervals**, **active sampling** to prioritize informative matchups, and **E-values** for sequential validity. The leaderboard reports CIs because top models routinely sit within overlapping intervals — rank order is partly statistical noise.
- **Rankings are fragile.** *Dropping Just a Handful of Preferences Can Change Top LLM Rankings* ([2508.11847](https://arxiv.org/abs/2508.11847)) — Bradley–Terry rankings can flip when a tiny worst-case fraction of votes is removed.
- **LLM-as-judge biases.** Position bias (favoring first/second option) and verbosity bias (favoring longer answers) — treat a judge as a measurement instrument with known biases, not an oracle.

---

## 8. Conformal prediction — distribution-free guarantees

- **Split conformal** (Vovk; Angelopoulos & Bates, [2107.07511](https://arxiv.org/abs/2107.07511)) — wrap any model, get prediction sets with finite-sample, distribution-free coverage ≥ 1−α, under exchangeability.
- **Under shift** — Tibshirani, Barber, Candès, Ramdas (2019) handles the realistic case where test ≠ train.
- **For generation** — Quach et al., *Conformal Language Modeling* (ICLR 2024) — output sets with correctness guarantees for LLM generation.
- **For evaluation** — conformal sets to quantify LLM-as-judge uncertainty ([2509.18658](https://arxiv.org/abs/2509.18658)); prediction-set width as a per-instance judge-reliability indicator.

---

## 9. Anytime-valid inference & e-values — a live statistics frontier

The classic test breaks if you peek; **sequential / always-valid** inference fixes it and is built for monitoring and online experimentation:

- **E-values** (Vovk & Wang) and **always-valid p-values / confidence sequences** (Howard, Ramdas et al.; Waudby-Smith & Ramdas) — inference that stays valid no matter when you stop or how often you look.
- **Why it's AI-relevant:** continuous monitoring of deployed models, drift detection, and online A/B experimentation all need anytime-valid guarantees. Chatbot Arena already uses E-values for sample-efficient sequential ranking.

---

## 10. Prediction-powered inference (PPI) — the method built for the AI era

The single most "modern AI era" statistical method.

- **The idea** (Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, *Science* 2023; [2301.09633](https://arxiv.org/abs/2301.09633)) — you have a few expensive gold-standard labels and a large pile of cheap ML predictions. PPI uses the predictions to shrink variance *while* using the gold labels to debias — yielding **valid confidence intervals and p-values** with the power of large ML-labeled data. **PPI++** tunes this for efficiency.
- **Why it matters:** the principled answer to "I labeled my dataset with an LLM — can I trust the resulting estimate?" Applied to LLM evaluations, GWAS, and more.
- Extensions to **E-values** ([2502.04294](https://arxiv.org/abs/2502.04294)) and **conformal/anytime-valid** settings ([2510.16166](https://arxiv.org/abs/2510.16166)) — uniting §8–§10.

**On the ground:** PPI is the tool for the now-ubiquitous workflow of *labeling data with a model and wanting honest error bars* — a skill almost no one has and everyone needs.

---

## 11. The pitfalls (statistical thinking as a shield)

The errors that quietly invalidate AI work: **data leakage** (test info in training — the most common eval killer), **benchmark contamination**, **Simpson's paradox**, **selection/survivorship bias**, **base-rate neglect**, and conflating **statistical vs. practical significance**. Most "surprising" results are one of these in disguise.

---

## How to use applied statistics at the frontier

The modern applied-stats toolkit you should be able to reach for: **bootstrap CIs + paired tests** for eval deltas (§7), **PPI** when you've labeled data with a model (§10), **conformal** for guaranteed prediction/eval intervals (§8), **anytime-valid tests** for monitoring and online experiments (§9), **calibration + proper scoring** for trustworthy probabilities (§6), and **DML/causal forests** for what an intervention actually does (§4).

**The immortal statistical core:** probability (distributions, expectation, CLT — *and its limits*); estimation & inference (likelihood, sampling distributions, bootstrap); experimental design & causal inference; regression and the bias-variance decomposition; and Bayesian inference. Sections 7–10 are where this core is *actively extending* into the AI era — which is precisely the frontier you want to stand on.
