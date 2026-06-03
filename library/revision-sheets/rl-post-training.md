# RL & Post-training — Revision Sheet

## The stack (in order)

```
Base model (pretrained)
    ↓ SFT
Instruction-following model
    ↓ DPO / RLHF
Helpful, harmless, honest assistant
    ↓ RLVR / GRPO (optional)
Reasoning model
```

## RLHF (the original)

1. Collect human preference pairs: (prompt, response_A, response_B, human_preference)
2. Train **reward model** R(x, y) to predict preferences
3. Fine-tune policy with **PPO** to maximize E[R(x, y)] - β·KL(π || π_ref)
   - KL term prevents the model from gaming the reward (reward hacking)
   
**Weakness:** reward models can be hacked, expensive, noisy

## DPO (the offline shortcut)

Observation: the reward model in RLHF can be expressed in closed form in terms of the optimal policy. So skip training a reward model entirely.

**Loss:** L_DPO = -log σ(β · [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])

where y_w = preferred response, y_l = rejected response.

**No RL loop, no reward model, no critic.** Just a classification-like loss on preference pairs.

**Variants:** KTO (unpaired), ORPO (no reference model), SimPO (length-normalized).

## GRPO (for reasoning)

For verifiable tasks (math, code), skip preference data entirely:

1. Sample G responses for each prompt: {o₁, ..., o_G}
2. Score each with rule-based reward (1/0 for correct/wrong)
3. **Group-relative advantage**: Aᵢ = (rᵢ - mean(r)) / std(r)
4. Update with PPO-style clipped objective using Aᵢ

**No critic, no reward model.** The group itself is the baseline.

## Key concepts

| Term | Meaning |
|------|---------|
| KL divergence penalty | Keeps policy close to reference; prevents reward hacking |
| Reward hacking / Goodhart's law | Optimizing the reward model proxy diverges from true intent |
| RLVR | RL with Verifiable Rewards — checkable signals (right/wrong) instead of preference models |
| Process reward model (PRM) | Rewards each reasoning step, not just final answer |
| Outcome reward model (ORM) | Rewards only final answer |
| RLAIF | AI (not human) provides the preference signal (Constitutional AI) |

## What emerged from DeepSeek-R1-Zero (no SFT, pure GRPO)

- Self-correction: the model spontaneously started reconsidering answers
- Reflection: explicitly checking its own work
- Longer chains of thought: more tokens = better answers on hard problems

None of these were explicitly trained. They emerged from the optimization pressure to get right answers.
