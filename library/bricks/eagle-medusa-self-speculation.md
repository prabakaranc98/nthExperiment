# EAGLE / Medusa Self-Speculation

**One-liner:** Speculative decoding without a separate draft model — bolt lightweight prediction heads onto the target model itself (Medusa) or a tiny autoregressive head that drafts in the target's own feature/hidden space (EAGLE), giving the highest acceptance rates of any spec-decode method in production engines.

## The two recipes

**Medusa (2024):** add K extra MLP heads on top of the last hidden state h_t, where head i predicts token t+i+1 in parallel. Decode forms a *tree* of candidate continuations from the top-k of each head; verify all paths in one forward pass via **tree attention** (a block-diagonal mask so each candidate only attends to its own prefix). Heads are cheap to train (frozen backbone, Medusa-1) and add minimal params.

**EAGLE (2024):** the key fix to Medusa's weakness — autoregress at the *feature* level, not the token level. The draft head is one small transformer layer that takes the target's previous-layer hidden state f and the embedding of the sampled token, and predicts the next feature f̂; the target's frozen LM head turns f̂ into a token distribution. Drafting is autoregressive but in feature space (which is more predictable), then verified by the target.

```
EAGLE draft step:  f̂_{t+1} = DraftLayer( concat(f_t, emb(token_t)) )
                   p_{t+1} = LMHead(f̂_{t+1})        # reuse target's head
verify:            one target forward over the drafted tree, accept longest matching prefix
```

EAGLE-2 (2024): dynamic, context-aware draft trees (expand high-confidence nodes). EAGLE-3 (2025): drop the feature-prediction loss, fuse multi-layer features, scale draft training → ~4-6x speedups, the current SOTA self-spec.

## Where it appears

- **vLLM / SGLang / TensorRT-LLM** — EAGLE/EAGLE-3 are the default shipping spec-decode path; no second model to host
- **Medusa (Cai et al.)** — used in production serving where training a draft model is impractical; tree verification kernel widely reused
- **Llama / DeepSeek deployments** — EAGLE heads trained per-model and served alongside the frozen backbone

## Common mistake

Thinking these change the output distribution. With the standard speculative-sampling acceptance rule, verification makes the output **distributionally identical** to plain target sampling — it's lossless, pure latency win. The other classic error: Medusa predicts *tokens* independently per head (positions are conditionally independent given h_t), which caps acceptance; EAGLE's gain comes precisely from making the draft *autoregressive in feature space* instead.

## See also
- [[speculative-decoding]] — the general draft-then-verify framework these specialize
- [[tree-token-tree-verification]] — the tree-attention mask that verifies many candidates in one pass
- [[multi-token-prediction]] — the training-time cousin (extra heads predicting future tokens)
