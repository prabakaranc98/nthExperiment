# Long Reasoning Chains (o1 / R1-style)

**One-liner:** Models RL-trained (on verifiable rewards) to emit very long internal CoT — exploring, backtracking, and self-verifying before answering — so accuracy scales smoothly with the thinking-token budget; the dominant frontier reasoning paradigm since late 2024.

## The key insight

Don't supervise the reasoning *trace* — supervise only the final *answer* with a verifiable reward (math answer correct, unit tests pass), then let RL discover long CoT on its own. The policy learns to spend more tokens when uncertain, producing emergent behaviors: enumerate cases, check work, notice errors ("Wait, that's wrong"), and backtrack. Objective is policy-gradient on outcome reward, typically GRPO-style:

J(θ) = E_{q, {oᵢ}~π_old} [ (1/G) Σᵢ min( rᵢ Âᵢ, clip(rᵢ, 1±ε) Âᵢ ) ] − β·KL(π_θ ‖ π_ref)

with rᵢ = π_θ(oᵢ|q)/π_old(oᵢ|q) and Âᵢ the group-normalized advantage (reward minus group mean over G sampled completions). Reward is outcome-only (RLVR); no learned reward model needed, which kills reward hacking.

Two orthogonal scaling axes: **train-time RL compute** and **test-time thinking tokens**. Accuracy on AIME/codeforces rises roughly log-linearly in the inference token budget.

## Where it appears

- **OpenAI o1 / o3 (2024–25)** — first deployed long-CoT models; "thinking" hidden from user, accuracy scales with reasoning effort (low/med/high).
- **DeepSeek-R1 (Jan 2025)** — open-weights; R1-Zero showed long CoT emerges from pure RLVR with *no* SFT cold-start, including spontaneous "aha" self-correction; R1 adds a small SFT cold-start for readability.
- **Kimi k1.5, Qwen3, Gemini 2.5 "thinking", Claude extended thinking** — same recipe, knobs for thinking budget.
- **Reasoning distillation** — long CoT from a big RL model SFT'd into small models, recovering most gains far cheaper than running RL on the small model.

## Common mistake

Believing the visible CoT is a faithful causal trace of the computation. It is RL-optimized for *getting the answer right*, not for being a transparent log — chains can be post-hoc rationalizations, and monitoring them is unreliable (CoT-faithfulness is an open safety problem). Also: more tokens ≠ monotonic gains — overlong chains overthink and can degrade on easy problems.

## See also
- [[rlvr]] — the outcome-only verifiable-reward signal that trains these models
- [[grpo]] — the group-relative policy-gradient algorithm behind R1
- [[budget-forcing-thinking-token-control]] — controlling the test-time thinking length
