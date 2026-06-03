# Causal ML

From the conceptual foundations of causality through causal representation learning and its intersection with modern ML.

---

## Conceptual foundations

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng: Causality (coming soon)](https://lilianweng.github.io) | Lilian Weng covers most frontier topics — check her blog for any new causal ML surveys | 🟡 |
| [Distill: Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) | Bayesian optimization as decision-making under uncertainty — the connection between probabilistic modeling and sequential decisions. Bridges to causal decision-making. | 🟡 |

---

## Course materials and lecture notes

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Brady Neal's Causal Inference Course](https://www.bradyneal.com/causal-inference-course) | The best free introductory causal inference course for ML people — covers potential outcomes, DAGs, identification, and ML-based estimators. | 🟡 |
| [Introduction to Causal Inference (online book)](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf) | Accompanying textbook to the course above. Rigorous but accessible. | 🟡 |
| [Causal Inference: The Mixtape (free online)](https://mixtape.scunning.com/) | Scott Cunningham's applied causal inference textbook — potential outcomes, IV, DiD, RDD, with real examples. The practitioner's guide. | 🟡 |

---

## Key survey papers (free)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Toward Causal Representation Learning (Schölkopf et al.)](https://arxiv.org/abs/2102.11107) | The agenda-setting paper for CRL — what it is, why it matters, the open problems. Required reading for the field. | 🔴 |
| [Invariant Risk Minimization (Arjovsky et al.)](https://arxiv.org/abs/1907.02893) | Learning representations that are invariant across environments — causality meets OOD generalization. The IRM paper. | 🔴 |
| [Double/Debiased Machine Learning (Chernozhukov et al.)](https://arxiv.org/abs/1608.00060) | Neyman-orthogonal estimators for causal effects using ML nuisance estimation. The rigorous applied causal inference method. | 🔴 |

---

## Toolkits (software + documentation)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [DoWhy documentation](https://www.pywhy.org/dowhy/) | Microsoft's causal inference library — identification + estimation + refutation, all in Python. Start here for implementation. | 🟡 |
| [EconML documentation](https://econml.azurewebsites.net/) | Microsoft's heterogeneous treatment effects library — DML, causal forests, metalearners. The applied causal ML toolkit. | 🟡 |
| [CausalML documentation](https://causalml.readthedocs.io/) | Uber's causal ML library — uplift modeling, treatment effect estimation. | 🟡 |

---

## The Distill / Bayesian connection

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Distill: A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) | GPs as distributions over functions — the Bayesian regression framework that underlies many causal estimation methods. | 🟡 |
| [Colah: Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | The information-theoretic framework that connects Bayesian inference, causality, and representation learning. | 🟢 |
