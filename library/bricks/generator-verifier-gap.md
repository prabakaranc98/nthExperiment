# Generator-Verifier Gap

**One-liner:** The asymmetry that checking a candidate solution is cheaper/more reliable than producing it — the conceptual reason test-time compute (best-of-N, search, self-correction) buys accuracy at all.

## The key insight

Let p_gen be the chance one sample is correct and let a verifier accept correct answers with probability T (true-positive rate) and accept wrong ones with probability F (false-positive rate). With N i.i.d. generations, *some* correct candidate exists with probability:

P(any correct) = 1 − (1 − p_gen)^N → 1 as N grows

That is the pass@N ceiling (oracle verifier). Real gain depends on the gap T ≫ F: a perfect verifier (T=1, F=0) makes best-of-N realize pass@N; a useless one (T≈F) collapses it to single-sample accuracy. The gap is large for problems in (loosely) NP-style classes — proofs, code that must pass tests, arithmetic, retrieval-grounded claims — where a certificate is checkable, and small for soft/subjective tasks where verification is as hard as generation.

## Where it appears

- **Best-of-N / rejection sampling** — generate N, score with a reward model or unit tests, keep the best; only helps when verifier ranking beats random
- **OpenAI o1/o3, DeepSeek-R1 (RLVR)** — RL with *verifiable* rewards (math/code checkers) trains long CoT precisely because the answer is cheap to verify
- **Process/outcome reward models (PRM/ORM), MCTS-style search, self-consistency** — all are verifier-guided ways to spend test-time compute
- **Self-correction / Reflexion** — the model acts as its own verifier; works only when its critique signal exceeds its generation noise

## Common mistake

Assuming the gap always exists, or that the verifier is free and perfect. If the verifier is itself learned and gameable, best-of-N becomes reward hacking — you optimize the proxy, not correctness. And for many open-ended tasks (essay quality, "is this aligned"), verification is *not* easier than generation, so test-time scaling flattens fast.

## See also
- [[rejection-sampling-best-of-n]] — the canonical mechanism that monetizes the gap
- [[rlvr]] — RL on verifiable rewards, the gap turned into a training signal
- [[reward-hacking-over-optimization]] — what happens when the verifier is weak/gameable
