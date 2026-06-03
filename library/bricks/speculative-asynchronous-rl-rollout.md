# Speculative / Asynchronous RL Rollout Infrastructure

**One-liner:** Decouple generation (rollout workers) from policy optimization (trainer) so GPUs never idle; correct for the resulting off-policy staleness with importance ratios — the systems pattern behind every large 2025 RLVR run.

## The problem it solves

Naive on-policy RL (PPO/GRPO) is synchronous: trainer waits for rollouts, rollout workers wait for the new policy weights. With long reasoning chains (10k–100k tokens) generation dominates wall-clock (often >70%), so the trainer's GPUs sit idle most of the step. Async decouples the two loops and runs them concurrently.

## The pattern

Two pools run in parallel:
- **Rollout/inference workers** (vLLM/SGLang) generate trajectories with policy weights θ_{behavior} that lag the trainer by k steps (the **staleness** / off-policy gap).
- **Trainer** consumes a replay buffer of those trajectories, updates to θ_{target}, and periodically pushes new weights back.

Off-policyness is corrected with the importance ratio inside the clipped surrogate (it is already there in PPO/GRPO — async just makes the denominator genuinely a *different* policy):

    ρ_t = π_θ(o_t | x, o_<t) / π_{behavior}(o_t | x, o_<t)
    L = E[ min(ρ_t · A_t, clip(ρ_t, 1−ε, 1+ε) · A_t) ]

Knobs: max staleness k (steps a sample may lag before it is dropped), generation/training ratio, and weight-sync cadence (NCCL broadcast or RDMA, often "in-flight" so workers swap weights mid-batch).

## Where it appears

- **AReaL / AReaL-boba** (Ant/Tsinghua, 2025) — fully async RLVR; decoupled rollout + trainer with staleness-aware data filtering, ~2–3× throughput over synchronous.
- **Magistral / Llama-Nemotron / Kimi-style RL** — production async pipelines feeding GRPO-family losses for long-CoT reasoning.
- **slime, OpenRLHF, verl (one-step-off / colocated-async)** — open frameworks exposing the gen/train ratio and weight-sync cadence as first-class config.
- "Speculative" rollouts — generate continuations under a stale/draft policy and accept under the current one, analogous to speculative decoding at the trajectory level.

## Common mistake

Letting staleness grow unbounded and treating the result as still on-policy. Once π_{behavior} drifts far from π_θ the importance ratios blow up (or saturate the clip), variance explodes, and training collapses. Async is only "free" while you cap k and clip/filter stale samples — it is fundamentally off-policy RL wearing a systems disguise.

## See also
- [[grpo]] — the loss most async rollout stacks actually optimize
- [[importance-sampling-the-off-policy-ratio]] — the correction that makes stale rollouts valid
- [[inference-and-serving]] — the vLLM/SGLang rollout workers are an inference-serving problem
