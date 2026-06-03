# 01 · Language Foundation Models

*The reference domain. Everything else is measured against what worked here.*

**The key insight that transferred everywhere:** pretraining on next-token prediction at scale produces representations that are surprisingly general. The token, the attention mechanism, and the scale recipe are the three things every other domain had to adapt.

## Landmark → Current Frontier

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Attention Is All You Need | Vaswani et al. | 2017 | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | The substrate |
| GPT-3 / Few-Shot Learners | Brown et al. | 2020 | [arXiv 2005.14165](https://arxiv.org/abs/2005.14165) | Scale → in-context learning |
| Chinchilla | Hoffmann et al. | 2022 | [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) | Compute-optimal training |
| The Llama 3 Herd | Dubey et al. | 2024 | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) | Complete open recipe |
| DeepSeek-R1 | DeepSeek-AI | 2025 | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | RL → reasoning |

**What language established as the default recipe:** transformer + BPE tokenization + next-token prediction + scaling + RLHF/DPO post-training. Every other domain starts by asking: "does this recipe work here, and what do I have to change?"
