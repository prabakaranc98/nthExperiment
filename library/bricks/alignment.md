# Alignment (Overview)

**One-liner:** The problem of making capable models reliably pursue intended goals/values — in 2024-26 practice a pipeline of preference learning (RLHF/DPO), AI-feedback scaling (RLAIF/Constitutional AI), and scalable oversight, fighting reward hacking, sycophancy, and deceptive generalization.

## The core objective

Post-training maximizes a reward (a proxy for human intent) under a KL leash to the SFT reference, so the model stays close to a competent base while shifting toward preferred behavior:

  max_θ  E_{x, y~π_θ}[ r(x,y) ] − β · KL(π_θ(·|x) ‖ π_ref(·|x))

- **r(x,y)** = reward model fit to human preferences via Bradley-Terry: P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l)).
- **β · KL** = the alignment tax / regularizer; too small → reward hacking + mode collapse, too large → no learning.
- **DPO insight:** the optimal π* of that objective has closed form, so you can skip the RM and optimize preferences directly (see [[dpo]]).

## The three layers

1. **Preference learning** — RLHF (PPO over a learned RM) or DPO/IPO/KTO (offline, RM-free). GRPO + verifiable rewards (RLVR) for reasoning.
2. **Scaling the supervision** — RLAIF / Constitutional AI: an AI critiques/revises against a written "constitution," replacing most human labels. Cuts cost; risks baking in the labeler model's blind spots.
3. **Scalable oversight** — supervising tasks humans can't directly grade: debate, recursive reward modeling, weak-to-strong generalization, process supervision (grade reasoning steps, not just answers).

## Where it appears

- **InstructGPT / RLHF** (Ouyang 2022) — SFT → RM → PPO; the template every chat model still follows.
- **Constitutional AI / RLAIF** (Bai 2022, Anthropic) — harmlessness from AI feedback + a principles list.
- **DPO** (Rafailov 2023) — preference optimization without RL; default for many open models (Llama/Tülu/Zephyr).
- **Weak-to-strong generalization** (OpenAI 2023) & **debate** — research bets on superhuman oversight.
- **Deliberative alignment / spec-driven** (o1-style, 2024-25) — train the model to reason over an explicit safety spec at inference.

## Common mistake

Conflating alignment with capability or with "refuses bad requests." Alignment is goal-faithfulness: a model can be highly capable, pass evals, and still be misaligned via **reward hacking** (optimizing the proxy r, not intent — Goodhart), **sycophancy** (telling raters what they want), or **deceptive alignment** (behaving aligned only while observed). Optimizing harder against an imperfect reward makes these worse, not better.

## See also
- [[rlhf]] — the canonical PPO-based instantiation of the objective above
- [[dpo]] — the RM-free closed-form alternative to the RL loop
- [[safety-evals]] — how you measure whether alignment held under stress
