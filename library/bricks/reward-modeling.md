# Reward Modeling (Bradley-Terry)

**One-liner:** Learn a scalar reward r_φ(x,y) from pairwise human preferences via a Bradley-Terry logistic loss so the chosen response scores above the rejected one — the learned proxy at the center of RLHF.

## The formula / definition

Bradley-Terry assumes the probability that response y_w beats y_l is a sigmoid of the reward gap:

P(y_w ≻ y_l | x) = σ( r_φ(x, y_w) − r_φ(x, y_l) )

Train by minimizing the negative log-likelihood over a preference dataset D = {(x, y_w, y_l)}:

L(φ) = − E_{(x,y_w,y_l)~D} [ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]

r_φ is usually a frozen-then-finetuned LLM with the LM head replaced by a scalar value head reading the last token's hidden state. Only **differences** in reward are identified — the absolute scale and an additive per-prompt constant are free.

## Where it appears

- **InstructGPT / RLHF** (Ouyang 2022) — the RM scores rollouts; PPO maximizes r_φ − β·KL to the reference policy.
- **DPO** (Rafailov 2023) — collapses the RM + RL into one loss; the implicit reward is r(x,y) = β·log(π_θ/π_ref) under the same Bradley-Terry likelihood, so no explicit RM is trained.
- **RLAIF / Constitutional AI** — same loss, preferences labeled by an LLM judge instead of humans.
- **Best-of-N / rejection sampling & reranking** — RM used purely as a scorer at inference, no policy update.

## Common mistake

Treating the RM scalar as a calibrated absolute quality score. It is only meaningful as a **difference**; the scale is arbitrary and prompt-relative. Over-optimizing against it (high KL from ref) drifts off the training distribution and triggers reward hacking — Goodhart's law, not a better policy.

## See also
- [[bradley-terry-model]] — the preference probability model the loss is derived from
- [[rlhf]] — the pipeline where this RM is the optimization target
- [[reward-hacking-over-optimization]] — what happens when the policy exploits the proxy
