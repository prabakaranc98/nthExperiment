# Agentic Reinforcement Learning (Tool / Agent RL)

**One-liner:** RL over multi-turn trajectories where the policy interleaves reasoning with real tool calls (search, code execution, browser) and is optimized against outcome/verifiable rewards on the final result — extending RLVR from single-turn answers to interactive, environment-grounded agents.

## The setup

The trajectory is a token sequence interleaving model tokens and **tool-response tokens** injected from the environment:

τ = (x, a₁, o₁, a₂, o₂, …, a_T, y)

- aₜ = model-generated action (reasoning + a tool call); oₜ = environment observation (search hits, stdout, errors)
- Reward r(τ) is usually **outcome-based and verifiable**: 1 if final y passes the checker (unit tests, math grader, exact-match), else 0. Optionally shaped with format/process terms.

Optimize with a group-relative policy gradient (GRPO-style), but **mask the loss on tool-response tokens** — you only train on tokens the policy actually produced:

Aᵢ = (rᵢ − mean(r)) / std(r)  over G sampled rollouts

L = E[ Σₜ∈model-tokens min(ρₜ·Aᵢ, clip(ρₜ,1−ε,1+ε)·Aᵢ) ],  ρₜ = π_θ(aₜ|·)/π_old(aₜ|·)

Multi-turn rollouts are slow and variable-length, so the rollout phase (long, environment-bound) is decoupled from the update phase, often via **partial/asynchronous rollouts**.

## Where it appears

- **DeepSeek-R1 → tool-augmented successors / Kimi k1.5** — outcome-reward RL extended to code execution and search loops
- **Search-R1 / ReTool / R1-Searcher** — RL teaches *when* to call search/code mid-reasoning, not just how; retrieved tokens masked from loss
- **OpenAI o3 / Deep Research, Claude with computer use** — frontier agents RL-trained to plan tool calls over long horizons against task success
- **SWE-bench-style coding agents (e.g. SWE-RL, Kimi-Dev)** — reward = tests pass; trajectory = edit→run→read-traceback loop

## Common mistake

Computing the policy-gradient loss over the *entire* trajectory including injected tool-output tokens. Those tokens weren't sampled from π_θ — backpropping through them makes the model try to "predict the environment," corrupting the gradient. Always mask observation tokens; only model-emitted actions carry loss.

## See also
- [[rlvr]] — the verifiable-reward signal that agentic RL extends to multi-turn settings
- [[grpo]] — the critic-free advantage estimator typically used for the policy update
- [[speculative-asynchronous-rl-rollout]] — decouples slow tool-bound rollouts from updates
