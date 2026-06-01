# Topic 12 · Post-Training

*Turning a base model into a capable assistant and reasoner: instruction tuning, RLHF/RLAIF/RLVR, and the preference-optimization family — read as estimation problems over preference and reward distributions.*

**Papers:** 20 · **Pace:** ~10 days at 2/day

---

## Paper Log

| # | Paper | Authors | Year | Status | Note | Blog | Exp |
|---|-------|---------|------|--------|------|------|-----|
| 1 | Finetuned Language Models Are Zero-Shot Learners (FLAN) | Wei et al. | 2021 | queued | Instruction tuning generalizes to unseen task types | — | — |
| 2 | Multitask Prompted Training (T0) | Sanh et al. | 2021 | queued | Explicit multitask prompting yields zero-shot generalization | — | — |
| 3 | Scaling Instruction-Finetuned LMs (Flan-T5) | Chung et al. | 2022 | queued | Scaling task count and CoT data in instruction tuning | — | — |
| 4 | Self-Instruct | Wang et al. | 2022 | queued | Bootstrapping instruction data from model's own generations | — | — |
| 5 | Learning to Summarize from Human Feedback | Stiennon et al. | 2020 | queued | Reward-model + PPO pipeline RLHF inherited | — | — |
| 6 | LIMA: Less Is More for Alignment | Zhou et al. | 2023 | queued | 1000 curated examples rival large-scale RLHF | — | — |
| 7 | The Unlocking Spell on Base LLMs (URIAL) | Lin et al. | 2023 | queued | Alignment shifts only a few token distributions | — | — |
| 8 | Zephyr: Direct Distillation of LM Alignment | Tunstall et al. | 2023 | queued | Distilled AI-feedback preferences + DPO as a cheap recipe | — | — |
| 9 | RLAIF | Lee et al. | 2023 | queued | AI-labeled preferences match human ones at scale | — | — |
| 10 | SLiC-HF | Zhao et al. | 2023 | queued | Sequence-likelihood calibration as contrastive alternative to RL | — | — |
| 11 | RRHF | Yuan et al. | 2023 | queued | Ranking responses to align without a PPO loop | — | — |
| 12 | Statistical Rejection Sampling Optimization (RSO) | Liu et al. | 2023 | queued | Sourcing preference pairs by rejection sampling | — | — |
| 13 | A General Theoretical Paradigm for Learning from Preferences (IPO) | Azar et al. | 2023 | queued | ΨPO framework; why DPO overfits near-deterministic preferences | — | — |
| 14 | KTO: Model Alignment as Prospect-Theoretic Optimization | Ethayarajh et al. | 2024 | queued | Human-utility loss on unpaired good/bad labels | — | — |
| 15 | ORPO: Monolithic Preference Optimization without Reference Model | Hong et al. | 2024 | queued | Odds-ratio penalty folds preference into SFT | — | — |
| 16 | SimPO | Meng et al. | 2024 | queued | Length-normalized, reference-free implicit reward | — | — |
| 17 | DeepSeekMath (GRPO) | Shao et al. | 2024 | queued | Group-relative, critic-free advantage estimation for reasoning RL | — | — |
| 18 | Tülu 3 | Lambert et al. | 2024 | queued | Fully open SFT → DPO → RLVR recipe | — | — |
| 19 | Secrets of RLHF (Part I) | Zheng et al. | 2023 | queued | Estimation and stability engineering underneath PPO-based RLHF | — | — |
| 20 | DAPO | Yu et al. | 2025 | queued | Decoupled-clip dynamic-sampling policy optimization for reasoning | — | — |

---

## Synthesis Notes
## Blog Post
## Experiments
