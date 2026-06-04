# Data Science & Statistics

Evaluation methodology, calibration, conformal prediction, and tabular foundation models — the statistical layer that determines whether results can be trusted.

**Difficulty scale:** 🟢 Accessible · 🟡 Intermediate · 🔴 Advanced. Cross-reference: [the concept-library index](../bricks/README.md) has reference cards for [Conformal Prediction], [Scaling Laws], calibration, and related concepts.

---

## Start here

The shortest path into the modern statistics-of-ML stack — read these three before anything else.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511) | Angelopoulos & Bates. The canonical, code-first tutorial — split conformal, RAPS, time-series, distribution shift, with Python notebooks. The one document to read on UQ. | 🟡 |
| [A Gentle Introduction (companion blog + code)](https://people.eecs.berkeley.edu/~angelopoulos/blog/posts/gentle-intro/) | The hands-on version of the tutorial above — runnable examples on real data; pair it with the paper. | 🟢 |
| [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | TabPFN v2 (Hollmann et al., *Nature* 2025). A single pretrained transformer beats tuned GBDTs on ≤10k-row tables in one forward pass — the result that reset tabular ML. | 🔴 |

---

## Key papers

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://arxiv.org/abs/2009.14193) | Angelopoulos et al. (ICLR 2021). The RAPS method — regularized adaptive prediction sets that stay small and stable. The practical conformal-classification recipe. | 🔴 |
| [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Guo et al. (2017). Shows modern nets are badly over-confident and that one-parameter temperature scaling fixes most of it. The calibration baseline everyone still cites. | 🟡 |
| [TabPFN (original)](https://arxiv.org/abs/2207.01848) | The first prior-data-fitted network for tables — in-context Bayesian prediction. Read for the mechanism before the Nature v2 results. | 🔴 |
| [From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting](https://arxiv.org/abs/2501.02945) | TabPFN-TS — treat forecasting as tabular regression with lightweight temporal features; no time-series-specific pretraining. How the tabular FM generalizes. | 🔴 |
| [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2506.16791) | Erickson et al. (NeurIPS 2025 Spotlight). A continuously-maintained leaderboard — GBDTs still strong, deep models catch up with ensembling, FMs win on small data. The reference benchmark now. | 🟡 |
| [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791) | Quantifies position, verbosity, and self-preference bias across 15 judge models. Read before trusting any LLM-as-judge eval number. | 🟡 |
| [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) | The structured overview of judge-model evaluation — methods, biases, and mitigation. The map of the now-dominant eval paradigm. | 🟡 |
| TabPFN-2.5 (Prior Labs, Nov 2025) | Scales the tabular FM toward larger tables and advances the TabArena state of the art. (No stable arXiv link yet — see the Prior Labs / TabPFN repo below.) | 🔴 |

---

## Blogs & explainers

Interactive and survey-level posts. Distill stopped publishing in 2021, but its visual explainers remain the best on their topics — kept as timeless classics.

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng: Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) | The taxonomy of LLM hallucination — factual vs. faithfulness failures and how to evaluate each. The clearest framing of LLM eval methodology. | 🟡 |
| [Lilian Weng: Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) | The complete contrastive-learning survey — SimCLR, MoCo, BYOL, CLIP, InfoNCE. Essential for understanding embeddings. | 🟡 |
| [Learning with not Enough Data, Part 1: Semi-Supervised Learning](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/) | Few labels: self-training, consistency regularization, pseudo-labeling. | 🟡 |
| [Learning with not Enough Data, Part 2: Active Learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/) | Query strategies for efficient labeling — uncertainty, diversity, and Bayesian sampling. | 🟡 |
| [Learning with not Enough Data, Part 3: Data Generation](https://lilianweng.github.io/posts/2022-04-15-data-gen/) | Synthetic data and augmentation for low-data settings. | 🟡 |
| [Distill: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) | t-SNE's hyperparameters and failure modes — what its clusters do and don't mean. Required before reading any t-SNE plot. | 🟢 |
| [Distill: A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) | GPs made interactive — priors over functions, kernels, posterior updates. | 🟡 |
| [Distill: Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) | How GPs power Bayesian optimization — acquisition functions, exploration-exploitation, hyperparameter tuning. | 🟡 |
| [Distill: A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) | The best visual GNN introduction — message passing, aggregation, pooling. | 🟢 |
| [Distill: Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/) | Spectral vs. spatial GNNs and what makes one expressive. | 🟡 |
| [Colah: Visualizing MNIST](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) | Dimensionality reduction on MNIST — the geometry of learned representations. | 🟢 |
| [Distill: AI Safety Needs Social Scientists](https://distill.pub/2019/safety-needs-social-scientists/) | Evaluation validity is a measurement-theory problem — construct validity applied to AI. Underread, still relevant. | 🟢 |

---

## Courses & talks

| Resource | What it covers | Level |
|----------|----------------|-------|
| [Awesome Conformal Prediction](https://github.com/valeman/awesome-conformal-prediction) | The continuously-updated hub for the field — 570+ papers, tutorials, videos, theses, and libraries, sorted by domain (time series, anomaly detection, LLMs). Where to go next on conformal prediction. | 🟡 |
| [Stanford HELM](https://crfm.stanford.edu/helm/) | Holistic Evaluation of Language Models — the standardized, transparent leaderboard for LLMs and VLMs across accuracy, robustness, bias, and efficiency. Browse HELM Lite for capability comparisons. | 🟡 |

---

## Tools

| Resource | What it is | Level |
|----------|-----------|-------|
| [TabPFN (Prior Labs)](https://github.com/PriorLabs/TabPFN) | The official tabular foundation model — pip-installable, scikit-learn API, classification + regression. The fastest way to try a tabular FM on your data. | 🟢 |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | scikit-learn-contrib conformal-prediction library — 20+ methods across regression, classification, time series, and risk control. The default Python entry point. | 🟢 |
| [TorchCP](https://github.com/ml-stat-Sustech/TorchCP) | PyTorch toolbox for conformal prediction on deep models — classification, regression, GNNs, LLMs, with GPU acceleration and CP-specific training. | 🟡 |
| [aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction) | The reference implementation accompanying the Gentle Introduction — minimal, readable conformal methods on real datasets. Read the code to learn the method. | 🟢 |
| [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | The de facto framework for few-shot LLM evaluation — 60+ benchmarks, hundreds of subtasks; backs the HF Open LLM Leaderboard. Standardize your eval here. | 🟡 |
