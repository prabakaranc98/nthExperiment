# Data Science & Statistics

Evaluation methodology, calibration, conformal prediction, and tabular foundation models — the statistical layer that determines whether results can be trusted.

---

## Evaluation as a discipline

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng: Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) | The taxonomy of LLM hallucination — factual errors, faithfulness failures, and how to evaluate them. Evaluation methodology for LLMs. | 🟡 |
| [Distill: AI Safety Needs Social Scientists](https://distill.pub/2019/safety-needs-social-scientists/) | Evaluation validity is a social science problem as much as a technical one — construct validity, measurement theory applied to AI. Underread. | 🟢 |

---

## Contrastive and representation learning (for DS)

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Lilian Weng: Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) | The complete contrastive learning survey — SimCLR, MoCo, BYOL, CLIP, InfoNCE. Essential for understanding embeddings. | 🟡 |

---

## Semi-supervised and data-efficient learning

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Learning with not Enough Data Part 1: Semi-Supervised Learning](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/) | When you have few labels: self-training, consistency regularization, pseudo-labeling. | 🟡 |
| [Learning with not Enough Data Part 2: Active Learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/) | Query strategies for efficient labeling — uncertainty sampling, diversity sampling, Bayesian approaches. | 🟡 |
| [Learning with not Enough Data Part 3: Data Generation](https://lilianweng.github.io/posts/2022-04-15-data-gen/) | Synthetic data, augmentation, and data generation for low-data settings. | 🟡 |

---

## Dimensionality reduction and visualization

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Distill: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) | t-SNE's hyperparameters, failure modes, and what its clusters actually mean (and don't mean). Required before using t-SNE on any result. | 🟢 |
| [Colah: Visualizing MNIST](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) | Dimensionality reduction applied to MNIST — the geometry of learned representations. | 🟢 |

---

## Gaussian processes and Bayesian optimization

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Distill: A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) | GPs made interactive and visual — priors over functions, kernel functions, posterior updates. | 🟡 |
| [Distill: Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) | How GPs power Bayesian optimization — acquisition functions, exploration-exploitation, hyperparameter tuning. | 🟡 |

---

## Graph-structured data

| Resource | Why read it | Level |
|----------|-------------|-------|
| [Distill: A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) | The best visual introduction to GNNs — message passing, aggregation, pooling, the design space. | 🟢 |
| [Distill: Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/) | The mathematical foundations of spectral vs. spatial GNNs — what makes a GNN expressive. | 🟡 |

---

## TabPFN and tabular foundation models

| Resource | Why read it | Level |
|----------|-------------|-------|
| [TabPFN paper](https://arxiv.org/abs/2207.01848) | The original TabPFN paper — in-context Bayesian prediction for tabular data, beating tuned GBDTs. | 🔴 |
| [TabPFN-v2 paper](https://arxiv.org/abs/2501.02945) | Scaling to 10k rows with mixed types — the production version. | 🔴 |
