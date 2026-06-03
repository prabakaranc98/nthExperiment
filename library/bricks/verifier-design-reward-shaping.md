# Verifier Design & Reward Shaping (RLVR)

**One-liner:** In verifiable-reward RL the checker *is* the alignment target, so engineering it — answer extraction/canonicalization, sandboxed unit-test harnesses, format gating, and filtering trivially-passing rollouts — determines what the policy actually learns far more than the RL algorithm does.

## The reward decomposition

A typical RLVR reward is a gated, shaped sum, not a single bit:

```
r(x, y) = 1[format_ok(y)] · ( w_v · verify(extract(y), gold)
                              + w_f · format_bonus(y)
                              − w_p · length_penalty(y) )
```

- `extract`: pull the candidate answer (\boxed{...}, ```code``` block, last line, `<answer>` tag).
- `canonicalize`: math-equivalence (sympy simplify, fraction/units normalization) so `1/2 == 0.5`; for code, normalize whitespace before running.
- `verify`: exact/symbolic match (math) or pass-all-tests (code), run in a resource/time-limited sandbox.
- `format_ok`: hard gate — malformed output scores ~0 regardless of correctness, which teaches parseable structure but risks reward hacking the format alone.

## The data-side trick (curriculum filtering)

Per prompt, estimate solve-rate p̂ from k rollouts and **drop prompts where p̂≈0 or p̂≈1** — these give zero advantage under GRPO's group-mean baseline (Adv_i = r_i − mean(r) → 0 when all rollouts agree). Keep medium-difficulty prompts where the gradient signal lives. This is dynamic sampling / difficulty filtering.

## Where it appears

- **DeepSeek-R1 / R1-Zero** — pure rule-based math+code verifiers plus a format reward (`<think>`/`<answer>` tags); deliberately *no* neural RM to avoid reward hacking.
- **Tülu 3 (RLVR)** — verifiable rewards for math/IFEval-style constraint following; binary correctness as the RL signal.
- **DAPO / Dr.GRPO** — dynamic sampling (drop p̂∈{0,1} groups), overlong-reward shaping, and removing length-normalization bias in the GRPO objective.
- **Code RL (SWE/competitive)** — hidden unit-test harnesses; reward = fraction (or all) tests passing, with timeout/OOM sandboxing.

## Common mistake

Treating the verifier as a fixed oracle and obsessing over the RL algorithm. The policy optimizes *exactly* what your checker measures: a brittle regex extractor makes correct answers look wrong (false negatives shrink the learning signal), while a loose checker is reward-hacked (false positives — `print(\"PASS\")`, answer-leaking in CoT, format-only rewards). Spec the verifier, audit its false-positive/false-negative rate, and red-team it before scaling RL.

## See also
- [[rlvr]] — the training paradigm this verifier sits inside
- [[reward-hacking-over-optimization]] — what a loose or gameable verifier produces
- [[grpo]] — group-baseline objective whose advantage signal motivates difficulty filtering
