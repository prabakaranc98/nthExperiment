# Tree / Token-Tree Verification

**One-liner:** Pack many candidate continuations as a token tree and verify them all in a single draft-model-free target forward pass via a custom (tree) attention mask, accepting the longest valid root-to-leaf path per step — the shared verification engine behind Medusa, EAGLE, and SpecInfer.

## The key insight

A linear draft of k tokens only ever explores one continuation. A tree explores many for nearly the same target-model cost, since target latency is dominated by weight loads, not the few extra tokens. Lay out the tree's N nodes in a flat sequence and feed them in one pass; a **tree attention mask** lets each node attend only to its ancestors (its prefix), so all root-to-leaf paths are scored simultaneously without cross-path leakage.

```
mask[i,j] = 1  iff  node j is an ancestor of node i (or j == i)   # else -inf
```

Verification per path (greedy/typical): walk root->leaf, accept token t_{d+1} iff it equals argmax (or passes the acceptance criterion) of the target logits computed at its parent node. The accepted prefix = the longest path all of whose tokens pass. Speculative-sampling acceptance (accept w.p. min(1, p_target/p_draft), else resample from the residual) generalizes the tree to exact-distribution sampling. Expected accepted length grows with tree width/depth but with diminishing returns; tuning the tree topology (which positions to branch) is the real lever.

## Where it appears

- **Medusa** — multiple decoding heads propose top-k per position; their Cartesian product is pruned to a fixed sparse tree, verified with tree attention.
- **EAGLE / EAGLE-2/3** — a lightweight autoregressive head drafts at the feature level; EAGLE-2 makes the tree *dynamic* (context-dependent depth/width via draft confidence) instead of a static template.
- **SpecInfer** — coined "token tree verification": merges multiple small draft models' outputs into one tree, verified in parallel against the LLM; core to serving speedups.
- **Sequoia, vLLM/TensorRT-LLM** — hardware-aware optimal tree sizing; tree spec-decode in production serving stacks.

## Common mistake

Thinking a bigger tree is always faster. Each extra node costs FLOPs and lengthens the attention mask; once the batch leaves the memory-bound regime (or many paths get rejected), wall-clock regresses. Also: forgetting that the mask must enforce strict ancestor-only visibility — a wrong mask silently lets siblings attend to each other and corrupts logits.

## See also
- [[speculative-decoding]] — the linear-draft parent algorithm tree verification generalizes
- [[eagle-medusa-self-speculation]] — the self-drafting heads that produce the trees
- [[kv-cache]] — tree nodes share ancestor KV; correct cache layout/rollback is what makes it cheap
