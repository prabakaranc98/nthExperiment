# Teacher Forcing & Exposure Bias

**One-liner:** Teacher forcing trains autoregressive models on ground-truth prefixes (each step conditioned on real previous tokens, not the model's own); exposure bias is the resulting train/inference distribution mismatch — at test time the model conditions on its own (possibly erroneous) generations, a regime it never saw during training.

## The definition / mechanism

Autoregressive factorization: p(y) = ∏ₜ p(yₜ | y₍<ₜ₎, x).

**Teacher forcing (training):** maximize log-likelihood conditioning on ground-truth context
  L = −∑ₜ log p(yₜ | y*₍<ₜ₎, x),   where y*₍<ₜ₎ are the *true* tokens.

**Inference (free-running):** condition on previously *generated* tokens
  ŷₜ ~ p(· | ŷ₍<ₜ₎, x).

Exposure bias = the model is only ever exposed to gold prefixes during training (states drawn from the data distribution), but at inference it visits states drawn from its own rollout distribution. Errors at step t shift the conditioning context off-manifold, and because the model never trained on those states, errors compound — the classic O(εT) → O(εT²) error-accumulation argument from imitation learning (DAgger / Ross & Bagnell, 2011).

## Where it appears

- **Scheduled sampling (Bengio et al., 2015)** — anneal a probability of feeding the model's own prediction instead of the gold token during training; the canonical "fix," though it yields a biased gradient estimator (Huszár, 2015).
- **LLM hallucination / snowballing** — exposure bias is the standard hand-wave for why long greedy generations drift, repeat, or compound early mistakes; degeneration (Holtzman et al., 2020) is the sampling-side counterpart.
- **RLHF / RLVR / GRPO** — sequence-level RL optimizes on *on-policy* model rollouts, eliminating the train/inference gap that MLE+teacher-forcing leaves; reward is computed on what the model actually generates.
- **Multi-token prediction & self-speculation (Medusa, EAGLE)** — train auxiliary heads/draft on model-consistent context to reduce reliance on pure next-gold-token supervision.
- **Discrete/masked diffusion & block-autoregressive models** — non-left-to-right factorizations partly sidestep strict prefix dependence.

## Common mistake

Believing exposure bias is *the* dominant cause of LLM failures, or that teacher forcing is "wrong." It is correct maximum-likelihood training, and at frontier scale with high-capacity models the empirical effect is far smaller than the imitation-learning theory suggests — sampling temperature, truncation, and data/coverage usually matter more. Scheduled sampling also does not give an unbiased likelihood gradient; it changes the objective.

## See also
- [[chain-of-thought]] — long autoregressive rollouts are exactly where compounding-error / drift concerns surface
- [[rlvr]] — on-policy RL trains on the model's own generations, directly closing the train/inference gap
- [[decoding-sampling-strategies]] — the inference-side knobs (temperature, top-p) that interact with exposure bias
