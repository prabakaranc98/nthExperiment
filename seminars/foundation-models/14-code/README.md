# 14 · Code Foundation Models

*The one domain with a free, unhackable reward: does it run, and do the tests pass?*

## The adaptation
- **The "token":** subword (BPE) tokens over interleaved code + natural language; some models add fill-in-the-middle (FIM) so the model can edit, not just append.
- **Pretraining objective:** next-token prediction on trillions of tokens of code, commits, issues, and docs — then post-training with **RL from verifiable rewards (RLVR)**.
- **Key inductive bias:** none baked into the architecture; the bias lives in the data (repo structure, FIM) and in the training loop (execution feedback), not in attention.
- **Verification signal:** test suites, compilers, type checkers, and runtime traces. Unlike most domains, correctness is *machine-decidable* — which is why reasoning RL pays off most here and in math.

## Landmark → Frontier
| Model / Paper | Org | Year | Link | Why it matters |
|---|---|---|---|---|
| AlphaCode (competition-level code) | DeepMind | 2022 | [Science](https://www.science.org/doi/10.1126/science.abq1158) | Massive sampling + test-based filtering reaches median competitive-programmer rank |
| HumanEval / Codex | OpenAI | 2021 | [arXiv 2107.03374](https://arxiv.org/abs/2107.03374) | First code FM at scale; defines `pass@k` and the canonical benchmark |
| Code Llama | Meta | 2023 | [arXiv 2308.12950](https://arxiv.org/abs/2308.12950) | Open code FM with long context + FIM; seeded the open ecosystem |
| StarCoder 2 | BigCode | 2024 | [arXiv 2402.19173](https://arxiv.org/abs/2402.19173) | Fully transparent data + weights; 600+ languages |
| SWE-bench | Princeton et al. | 2023 | [arXiv 2310.06770](https://arxiv.org/abs/2310.06770) | Shifts evaluation from snippets to *resolving real GitHub issues* |
| SWE-bench Verified | OpenAI | 2024 | [openai.com](https://openai.com/index/introducing-swe-bench-verified/) | Human-vetted 500-task subset; the de-facto agentic-coding scoreboard |
| SWE-agent | Princeton | 2024 | [arXiv 2405.15793](https://arxiv.org/abs/2405.15793) | Shows the *agent-computer interface* drives performance as much as the model |
| DeepSeek-Coder-V2 | DeepSeek | 2024 | [arXiv 2406.11931](https://arxiv.org/abs/2406.11931) | Open MoE code model competitive with closed frontier on coding + math |
| Qwen2.5-Coder | Alibaba | 2024 | [arXiv 2409.12186](https://arxiv.org/abs/2409.12186) | Strong open code FM; repair, FIM, long context |
| DeepSeek-V3 | DeepSeek | 2024 | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) | 671B MoE base; the substrate beneath R1's coding gains |
| DeepSeek-R1 | DeepSeek | 2025 | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | RLVR with no SFT cold-start → emergent reasoning; largest wins on code/math |
| Kimi K2 | Moonshot AI | 2025 | [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) | 1T-param open agentic model trained with large-scale tool-use synthesis |
| Qwen3 | Alibaba | 2025 | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) | Hybrid think/no-think frontier; basis for the Qwen3-Coder agent line |

## Where it stands (2025-2026)
- **Agents, not autocompletion.** The unit of work is now "resolve this issue end-to-end" — read repo, edit, run tests, iterate. SWE-bench Verified is the headline metric; see the [live leaderboard](https://www.swebench.com/verified.html).
- **Open weights caught the frontier.** DeepSeek (V3/R1), Qwen3-Coder, and Kimi K2 are within striking distance of the best closed models on agentic coding, mostly under permissive licenses.
- **Reasoning RL is standard.** Post-R1, frontier coding models train against executable environments (compile/run/test loops), not just static labels.
- **Benchmarks are saturating and leaking.** Verified scores cluster in the high 80s+; private/contamination-resistant suites (SWE-bench Pro, long-horizon evolution and feature-build benchmarks) now reveal the real gaps that Verified hides.

## Transferred vs. reinvented
**Transferred directly from language modeling**
- Decoder-only Transformer, BPE tokenization, next-token pretraining.
- Scaling laws, MoE for cheap capacity, long-context attention.
- RLHF machinery — repurposed into RLVR.

**Had to be invented here**
- Fill-in-the-middle objectives for editing, not just left-to-right generation.
- Repo-level context construction (dependency-aware packing) over file-at-a-time.
- The **agent-computer interface**: tools, shell, file-edit primitives, and feedback loops.
- Verifiable-reward harnesses: sandboxed execution, test runners, and unhackable scoring.

## Open problems
- **Reward hacking.** Models learn to pass tests without solving the task (overfit patches, deleted tests, hardcoded outputs); verification must get stricter than the proxy.
- **Long-horizon work.** Multi-file refactors and multi-day features still break: context drift, lost state, compounding errors.
- **Eval integrity.** Training contamination and flawed/unsolvable test cases inflate headline scores; trustworthy measurement is unsolved.
- **Beyond pass/fail.** Tests verify behavior, not security, performance, maintainability, or intent — the rewards that matter most are the hardest to make verifiable.

## See also
- [11 · Design Patterns](../11-design-patterns/README.md) — RLVR, agents, and verification as cross-domain primitives
- [01 · Language](../01-language/README.md) — the base models code FMs are built on
- [10 · Multimodal](../10-multimodal/README.md) — code as one modality in tool-using agents
- [Concept library: bricks](../../../library/bricks/README.md) — RLVR, scaling laws, conformal/verification notes
