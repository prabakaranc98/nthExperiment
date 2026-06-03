# Length Normalization / Bias Control

**One-liner:** Counteract the systematic bias of preference and RL objectives toward longer outputs — because sequence log-probs and reward models both correlate reward with length — via length-normalized rewards (SimPO), explicit length penalties, or length-debiased advantages.

## Why the bias exists

Two compounding sources:
1. **Implicit reward scales with length.** In DPO, the implicit reward is β·log(π_θ(y|x)/π_ref(y|x)) = β·Σ_t log-ratio. The sum grows with |y|, so the optimizer can raise reward just by emitting more tokens.
2. **Reward models prefer long answers.** Human/RM preference data has a spurious length–quality correlation; RLHF then over-optimizes it (a form of reward hacking → "verbosity collapse").

## The fixes

**SimPO (Meng et al., 2024)** — drop the reference model, use the *average* log-prob as implicit reward (length-normalized), plus a target margin γ:

  r(x,y) = (β/|y|) · Σ_t log π_θ(y_t | x, y_<t)
  L = −log σ( r(x,y_w) − r(x,y_l) − γ )

Dividing by |y| removes the per-token length advantage and aligns the training objective with length-normalized generation.

**R-DPO / length-regularized DPO** — add an explicit penalty α·(|y_w| − |y_l|) inside the logistic to debias the preference term.

**RL (PPO/GRPO) length control** — subtract a length penalty from reward, or length-normalize/standardize advantages so longer rollouts don't get inflated returns; GRPO's mean-token loss vs sequence-sum loss choice directly changes the length bias (Dr. GRPO fixes a length-bias term in the normalization).

## Where it appears

- **SimPO** — reference-free preference tuning; length-normalized reward is the core trick, often matches/beats DPO with shorter outputs.
- **RLHF reward modeling** — adding a length penalty or training a length-disentangled RM (e.g., ODIN) to stop the RM rewarding verbosity.
- **GRPO / Dr. GRPO (2025)** — removing the response-length normalization that biased advantage estimates in long-CoT RLVR training.
- **AlpacaEval 2.0 LC** — length-controlled win rate, a regression-based debias of judge length bias at *eval* time.

## Common mistake

Treating it purely as a generation/decoding nuisance fixable with a length penalty at inference. The bias is baked into the *training objective*: an un-normalized DPO/RL gradient actively rewards length, so you must debias the loss (normalize or penalize during training), not just post-hoc clip outputs.

## See also
- [[dpo]] — the un-normalized implicit reward is the source of the length bias
- [[dpo-variants]] — SimPO, R-DPO and other length-aware preference objectives
- [[reward-hacking-over-optimization]] — verbosity is a canonical reward-hacking failure mode
