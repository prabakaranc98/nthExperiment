# 14 · Code Foundation Models

*Code generation, execution-grounded reasoning, and the verified-reward advantage.*

**Why code is special among FM domains:** code has a perfect verifiability signal — it either runs, passes tests, or produces correct output. This makes RL with verifiable rewards (RLVR) especially powerful here, and it's the domain where reasoning models (R1, o1) have shown the clearest wins.

**The "token":** code tokens via BPE. The pretraining: next-token prediction on code + text. The key differentiator: test execution as a reward signal.

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| Evaluating LLMs Trained on Code (HumanEval) | Chen et al., OpenAI | 2021 | [arXiv 2107.03374](https://arxiv.org/abs/2107.03374) | Codex + the HumanEval benchmark; pass@k estimator |
| Code Llama | Rozière et al., Meta | 2023 | [arXiv 2308.12950](https://arxiv.org/abs/2308.12950) | Open code FM based on Llama 2 |
| StarCoder 2 | Li et al., BigCode | 2024 | [arXiv 2402.19173](https://arxiv.org/abs/2402.19173) | Transparent open code model; 600+ programming languages |
| DeepSeek-Coder-V2 | DeepSeek | 2024 | [arXiv 2406.11931](https://arxiv.org/abs/2406.11931) | MoE code model; strong on competitive programming |
| AlphaCode 2 (competition-level code) | Li et al., DeepMind | 2023 | [Science 2022](https://www.science.org/doi/10.1126/science.abq1158) | Massive sampling + filtering at competition level |
| SWE-bench | Jimenez et al. | 2023 | [arXiv 2310.06770](https://arxiv.org/abs/2310.06770) | Resolving real GitHub issues; the agentic coding benchmark |
| SWE-agent | Yang et al. | 2024 | [arXiv 2405.15793](https://arxiv.org/abs/2405.15793) | Agent-computer interface governs performance |
| Qwen2.5-Coder | Hui et al., Alibaba | 2024 | [arXiv 2409.12186](https://arxiv.org/abs/2409.12186) | SOTA open code FM; strong on code repair and math |
| DeepSeek-R1 (coding) | DeepSeek | 2025 | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | RL with verifiable rewards → reasoning on code |

**The verifiable-reward advantage:** code is the domain that most clearly shows why RLVR works — "does this code pass the test suite?" is a free, unhackable reward signal. The improvements from reasoning RL (R1, o1) are largest on code and math precisely because of this.
