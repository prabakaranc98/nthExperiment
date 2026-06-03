# Test-Time Compute Scaling

**One-liner:** Inference compute — more samples, longer chains, or search — is a separate scaling axis that predictably lowers error, and per the compute-optimal frontier it is often cheaper to spend FLOPs at test time than to grow the model.

## The key insight

There are two knobs for a fixed task: model size N (set at train time) and inference compute T (set per query). Accuracy improves as a power law in T across multiple mechanisms:

- **Sequential** — longer CoT / "thinking" tokens. Error decays roughly with chain length until it saturates.
- **Parallel** — draw k samples, aggregate. Best-of-N with a verifier, or self-consistency (majority vote): accuracy(k) climbs with diminishing returns. `pass@k = 1 − (1−p)^k` is the *oracle* upper bound, not what a verifier achieves.
- **Search** — beam / tree / MCTS over reasoning steps scored by a process reward model (PRM).

Compute-optimal allocation (Snell et al., 2024): the best split between N and T depends on difficulty. On easy/medium problems, scaling T on a small model beats a 14x-larger model at matched FLOPs; on the hardest problems, more pretraining still wins. So `argmin_error subject to FLOPs = c·N·T_pretrain + (inference FLOPs)` is problem-dependent, not a single curve.

## Where it appears

- **OpenAI o1/o3, DeepSeek-R1, Gemini "thinking"** — RL-trained long CoT; o1's report shows accuracy rising log-linearly in both train-time RL and test-time thinking compute.
- **AlphaCode / AlphaGeometry** — massive parallel sampling + filtering/verification as the core engine.
- **Self-consistency (Wang et al.) & Best-of-N** — the cheapest parallel scaling; gains gated by the generator-verifier gap.
- **s1 / budget forcing (2025)** — control test-time compute by injecting/suppressing "Wait" tokens to extend or truncate thinking.

## Common mistake

Assuming test-time compute scales without bound. Sequential scaling saturates (and over-long chains can *degrade* via overthinking); majority-vote parallel scaling is capped by the model's coverage and the verifier's accuracy, not by `pass@k`. More samples cannot recover an answer the model never generates.

## See also
- [[compute-optimal-test-time-allocation]] — the formal train-vs-test FLOP tradeoff per difficulty
- [[pass-k-self-consistency-estimation]] — the parallel-sampling aggregation and its oracle bound
- [[long-reasoning-chains]] — the sequential-scaling mechanism and its failure modes
