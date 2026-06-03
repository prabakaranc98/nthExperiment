# Tree-of-Thought (ToT)

**One-liner:** Reframe reasoning as deliberate search over a tree of partial "thought" states — generate candidate next-thoughts, score them with a value heuristic, and explore via BFS/DFS with pruning and backtracking, replacing flat CoT's single left-to-right rollout.

## The framework (Yao et al., 2023)

A problem is decomposed into intermediate **thought** steps; state s = partial solution (the path so far). ToT instantiates four choices:

1. **Thought decomposition** — what is one step (a line, an equation, a sentence).
2. **Thought generator** G(s) → {s'} — sample k thoughts i.i.d. ("propose" prompt) or enumerate distinct continuations.
3. **State evaluator** V(s) → scalar/class — the LM rates each state ("value" 0–10) or **votes** across sibling states to rank them.
4. **Search algorithm** over the tree:
   - **BFS:** keep the top-b states per level (beam over thoughts).
   - **DFS:** descend most-promising child; backtrack when V(s) < threshold (dead-end pruning).

Pseudocode (BFS):
```
S0 = {root}
for step t in 1..T:
    Sgen = { s + g : s in S_{t-1}, g in G(s) }   # expand
    V = evaluate(Sgen)                            # LM scores/votes
    S_t = argtop_b(V, Sgen)                       # prune to beam b
return best leaf in S_T
```
Key knobs: branching factor k, beam width b, depth T → search budget ~k·b·T LM calls.

## Where it appears

- **Yao et al., "Tree of Thoughts" (NeurIPS 2023)** — Game of 24, creative writing, mini crosswords; ToT solves 74% of Game-of-24 vs ~4% for CoT.
- **Graph-of-Thoughts / Reasoning-as-Planning** — generalize the tree to DAGs and explicit world-model rollouts.
- **MCTS-style inference search & test-time scaling** — ToT is the heuristic-search precursor to UCT-guided LM search and o1/R1-style deliberate reasoning; the value evaluator becomes a learned PRM.

## Common mistake

Conflating ToT with self-consistency. Self-consistency samples N independent flat CoT chains and majority-votes the *final answers* — no intermediate evaluation, no pruning, no backtracking. ToT evaluates and prunes *partial* states mid-search and can abandon dead branches. It is structured search, not parallel sampling.

## See also
- [[chain-of-thought]] — the flat, single-path baseline ToT generalizes into a search tree
- [[mcts-style-inference-search]] — replaces ToT's hand-tuned BFS/DFS heuristics with principled UCT tree search
- [[self-consistency]] — the parallel-sampling alternative often confused with ToT
