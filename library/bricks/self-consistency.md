# Self-Consistency

**One-liner:** Sample K chain-of-thought traces at nonzero temperature, then majority-vote over the *final answers* (marginalizing out the reasoning paths) instead of taking one greedy chain — the canonical parallel test-time-compute baseline.

## The formula / definition

Sample K traces (r_i, a_i) ~ p(· | prompt) with temperature T > 0 (typ. 0.5-0.8, top-p ~0.9). The reasoning path r_i is a latent; the prediction marginalizes over it by voting on the answer:

    â = argmax_a  Σ_{i=1}^{K} 1[a_i = a]

Optionally weighted: â = argmax_a Σ_i w_i · 1[a_i = a], with w_i a normalized trace probability or a verifier/RM score (weighted SC). Requires answers be *comparable* — extract a canonical final answer (number, MC letter, normalized string). Accuracy rises monotonically with K then saturates; this is the "vote@K" curve, distinct from pass@K (which credits any-correct, not the majority).

## Where it appears

- Wang et al. 2022 (Self-Consistency Improves CoT) — original; +10-18 pts on GSM8K/SVAMP over greedy CoT.
- Reasoning models (o1/o3, DeepSeek-R1, Gemini-thinking) — "cons@64 / maj@K" reported alongside pass@1 as the parallel-scaling test-time-compute axis.
- Compute-optimal test-time scaling — SC is the embarrassingly-parallel baseline that sequential methods (revisions, search) and best-of-N+verifier are measured against.
- Universal Self-Consistency / answer-clustering variants — use an LLM to cluster free-form (non-extractable) answers before voting.

## Common mistake

Confusing vote@K (majority answer — what SC actually does) with pass@K (an *oracle* upper bound that any sample is correct). Pass@K only bounds SC; reporting pass@K as if it were achievable without a verifier overstates real accuracy. Also: SC only helps when the correct answer is the *modal* one — on hard problems where errors are correlated/systematic, the majority can be confidently wrong, and it cannot exceed what's in the sample distribution.

## See also
- [[chain-of-thought]] — SC samples and votes over diverse CoT traces
- [[rejection-sampling-best-of-n]] — sibling parallel method; picks via verifier/RM instead of plurality vote
- [[test-time-compute-scaling]] — SC is the canonical parallel-sampling axis of test-time compute
