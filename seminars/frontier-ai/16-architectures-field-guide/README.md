# Topic 16 · Architectures — From CNNs to Hybrids

*The model-design lineage in mechanical detail: convolutional and recurrent ancestors, the transformer family tree, efficiency and linear-attention zoo, sparse mixtures, and SSM–attention hybrids. The field guide threading under Topics 1, 4, 5, and 6.*

**Papers:** 20 · **Pace:** ~10 days at 2/day · *Read whenever you want to get inside the boxes other topics treat as given.*

---

## Paper Log

| # | Paper | Authors | Year | Status | Note | Blog | Exp |
|---|-------|---------|------|--------|------|------|-----|
| 1 | ImageNet Classification with Deep CNNs (AlexNet) | Krizhevsky, Sutskever & Hinton | 2012 | queued | GPUs + ReLU + dropout; the result that opened the era | — | — |
| 2 | Very Deep Convolutional Networks (VGG) | Simonyan & Zisserman | 2014 | queued | Depth from stacked 3×3 convolutions | — | — |
| 3 | Going Deeper with Convolutions (GoogLeNet / Inception) | Szegedy et al. | 2014 | queued | Multi-scale modules and parameter efficiency | — | — |
| 4 | Long Short-Term Memory (LSTM) | Hochreiter & Schmidhuber | 1997 | queued | Gated recurrence as fix for vanishing gradients | — | — |
| 5 | Sequence to Sequence Learning with Neural Networks | Sutskever, Vinyals & Le | 2014 | queued | The encoder–decoder bottleneck that attention relieved | — | — |
| 6 | A ConvNet for the 2020s (ConvNeXt) | Liu et al. | 2022 | queued | Modernized ConvNet matching ViTs — controlled ablation | — | — |
| 7 | BERT | Devlin et al. | 2018 | queued | Encoder-only, bidirectional masked LM | — | — |
| 8 | Language Models Are Unsupervised Multitask Learners (GPT-2) | Radford et al. | 2019 | queued | Decoder-only scaling thesis + byte-level BPE | — | — |
| 9 | ELECTRA | Clark et al. | 2020 | queued | Replaced-token detection — far more sample-efficient pretraining | — | — |
| 10 | Swin Transformer | Liu et al. | 2021 | queued | Hierarchical windowed attention; general-purpose vision backbone | — | — |
| 11 | Perceiver IO | Jaegle et al. | 2021 | queued | Cross-attention to fixed latent array — arbitrary modalities | — | — |
| 12 | Efficient Transformers: A Survey | Tay et al. | 2020 | queued | Map of the efficiency zoo | — | — |
| 13 | Transformers are RNNs (Linear Attention) | Katharopoulos et al. | 2020 | queued | Kernelized attention = linear-time autoregressive recurrence | — | — |
| 14 | Rethinking Attention with Performers | Choromanski et al. | 2020 | queued | FAVOR+ random features for unbiased softmax-attention estimator | — | — |
| 15 | Hyena Hierarchy | Poli et al. | 2023 | queued | Long implicit convolutions + gating as subquadratic attention replacement | — | — |
| 16 | RWKV: Reinventing RNNs for the Transformer Era | Peng et al. | 2023 | queued | Attention-free, parallel-trainable RNN with constant-memory inference | — | — |
| 17 | Retentive Network (RetNet) | Sun et al. | 2023 | queued | Retention with parallel, recurrent, and chunkwise forms | — | — |
| 18 | Outrageously Large Neural Networks (Sparsely-Gated MoE) | Shazeer et al. | 2017 | queued | Conditional-computation MoE layer — origin of sparse scaling | — | — |
| 19 | Switch Transformers | Fedus, Zoph & Shazeer | 2021 | queued | Top-1 routing simplifying MoE to trillion-parameter scale | — | — |
| 20 | Jamba | Lieber et al. | 2024 | queued | Interleaved Mamba and attention with MoE — hybrid SSM–transformer | — | — |

---

## Synthesis Notes
## Blog Post
## Experiments
