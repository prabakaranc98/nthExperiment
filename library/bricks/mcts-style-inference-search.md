# MCTS-Style Inference Search (LATS)

**One-liner:** Run Monte Carlo Tree Search over reasoning/action steps — selection → expansion → simulation → backup — using an LLM as the policy and a value model/PRM as the evaluator, the "AlphaZero-for-reasoning" framing that PRMs exist to guide.

## The four phases (one MCTS iteration)

Each node = a partial reasoning trajectory (sequence of steps/actions). Repeat:

1. **Selection** — descend from root by UCT, picking the child maximizing:
   `UCT(s,a) = Q(s,a) + c · P(a|s) · √(N(s)) / (1 + N(s,a))`
   where Q is the mean backed-up value, N visit counts, P the LLM's prior over the next step (PUCT/AlphaZero variant).
2. **Expansion** — sample k candidate next steps from the LLM policy at the leaf.
3. **Simulation/Evaluation** — score the leaf with a **value model or PRM** (and/or roll out to a terminal answer + outcome reward / self-reflection). LATS adds an environment-feedback + self-reflection signal.
4. **Backup** — propagate the value up the path: `N(s,a) += 1`, `Q(s,a) += (v − Q(s,a)) / N(s,a)`.

Final answer = best child of root (highest Q or visits), or majority over high-value leaves.

## Where it appears

- **LATS (Zhou et al., 2024)** — "Language Agent Tree Search": MCTS over ReAct-style action trees, LLM as policy + value, environment observations + self-reflection as the reward signal for agentic tasks.
- **AlphaZero-for-reasoning lineage** — TS-LLM, ReST-MCTS\*, rStar/rStar-Math — MCTS to generate high-value reasoning traces, often used to *train* a PRM and bootstrap the policy.
- **PRM-guided search** — the canonical consumer of process reward models; PRM supplies the per-step value that makes tree search beat flat best-of-N at matched compute.

## Common mistake

Believing MCTS strictly dominates simpler test-time search. With a weak/miscalibrated value model, MCTS often loses to best-of-N or beam search at equal FLOPs — and an exploitable PRM gets reward-hacked (the search drives toward high-PRM, wrong-answer branches). The tree only pays off when the verifier signal is strong and the branching factor is well-controlled.

## See also
- [[process-vs-outcome-reward-models]] — the PRM is the value/evaluator that guides selection and backup
- [[tree-of-thought]] — the non-MCTS tree-search ancestor (BFS/DFS over thoughts, no UCT/backup)
- [[compute-optimal-test-time-allocation]] — when MCTS beats best-of-N / beam search at fixed compute
