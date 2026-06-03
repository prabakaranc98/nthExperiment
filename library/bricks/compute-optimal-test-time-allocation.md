# Compute-Optimal Test-Time Allocation

**One-liner:** Given a fixed inference budget, pick the test-time strategy (more parallel samples vs. deeper sequential revision vs. tree search) and how hard to think *per prompt by difficulty* to maximize accuracy per FLOP — often beating a 14x larger model on the same total compute.

## The key insight (Snell et al., 2024)

Frame it as a budget-constrained optimization. For a question q with budget N (generations):

θ*(q, N) = argmax_θ  E[ correct | strategy θ, budget N, q ]

The optimal strategy is **difficulty-adaptive**: easy prompts → spend little (best-of-1 or short CoT); hard prompts → spend more, and *shift* allocation. Empirically:
- **Easy/medium:** sequential revision (self-correct, refine one chain) wins — narrow, deep edits.
- **Hard:** parallel sampling + verifier search (best-of-N, beam over a PRM) wins — broad coverage.

Optimal allocation ratio between sequential and parallel shifts with difficulty; using the per-prompt-optimal mix gave ~4x less compute than naive best-of-N at matched accuracy, and let a small model outperform a ~14x larger one under FLOP-matched comparison.

The cross-cutting tradeoff: **train-time vs. test-time compute are substitutes** — you can move FLOPs from pretraining a bigger model into thinking longer at inference, up to a difficulty-dependent crossover (hard questions still favor the bigger base model).

## Where it appears

- **Snell et al. (2024), "Scaling Test-Time Compute Optimally"** — the original difficulty-binned allocation result; PRM-guided beam search vs. revision.
- **o1 / o3 / R1-style reasoning models (2024-2025)** — RL'd to emit long CoT; serving knobs trade latency for accuracy along an inference scaling curve.
- **DeepMind / Large Language Monkeys (Brown et al., 2024)** — coverage (pass@k) scales as a power law in samples; allocation = where on that curve to sit.
- **Inference scaling laws** — accuracy vs. log(test-time FLOPs) is roughly linear over a regime, mirroring pretraining scaling laws.

## Common mistake

Treating "more test-time compute" as a single monotone dial. The *right strategy flips with difficulty and with verifier quality* — best-of-N helps only as far as your verifier/aggregator can pick the right answer (generator-verifier gap). Past a point, extra samples just inflate FLOPs while a weak verifier caps achievable accuracy; and easy questions waste budget under a search-heavy policy.

## See also
- [[test-time-compute-scaling]] — the broader scaling phenomenon this allocates within
- [[mcts-style-inference-search]] — the deep-search strategy on one end of the allocation spectrum
- [[generator-verifier-gap]] — why best-of-N gains saturate, bounding the parallel-sampling arm
