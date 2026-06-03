# RLVR (RL with Verifiable Rewards)

**One-liner:** Replace the learned reward model with a programmatic verifier (math answer-checking, unit tests, format/regex checks) returning a ground-truth-grounded, hard-to-hack scalar reward — the o1/R1-style reasoning recipe that scales RL on domains with checkable answers.

## The setup

Policy πθ generates a full chain-of-thought + answer y for prompt x. Reward is computed by a deterministic verifier, not a neural RM:

r(x, y) = 1 if verify(x, extract_answer(y)) else 0   (typically binary, outcome-level)

- **Math:** parse boxed answer, normalize, check equality (often via symbolic/SymPy match) vs ground truth.
- **Code:** run y against a hidden unit-test suite; r = fraction passing (or 1 iff all pass).
- **Format shaping:** small bonus for correct `<think>...</think>` / `\boxed{}` structure.

Optimize with GRPO (group-relative, no value net) or PPO. GRPO advantage for sample i in a group of G rollouts on the same x:

Âᵢ = (rᵢ − mean(r₁..r_G)) / std(r₁..r_G)

plus a KL penalty β·KL(πθ ‖ π_ref) to the SFT base. Because r is grounded in truth, there is no reward model to over-optimize against — the classic Goodhart failure mode of RLHF is largely removed.

## Where it appears

- **DeepSeek-R1 / R1-Zero (2025)** — RLVR with GRPO directly on math+code from a base model (R1-Zero needs *no* SFT); long CoT and self-verification emerge purely from verifiable reward.
- **OpenAI o1/o3** — large-scale RL on reasoning with checkable rewards; test-time-compute scaling.
- **Tülu 3 (AI2)** — names and popularizes "RLVR" as a distinct post-training stage alongside SFT and DPO.
- **Open repros** — TRL/verl/OpenRLHF, Open-Reasoner-Zero, SkyThought; math (GSM8K/MATH/AIME) and code (LiveCodeBench).

## Common mistake

Assuming the binary verifier reward is unhackable. The *answer* is hard to fake, but the verifier and pipeline aren't: brittle answer-extraction regexes, weak/incomplete unit tests, reward leakage (the answer appearing in the prompt), and "format reward" gaming all let the policy collect reward without genuine reasoning. RLVR moves the attack surface from the reward model to the verifier — design it adversarially.

## See also
- [[grpo]] — the group-relative, critic-free RL algorithm RLVR is usually trained with
- [[verifier-design-reward-shaping]] — building robust verifiers is where RLVR succeeds or fails
- [[reward-hacking-over-optimization]] — RLVR mitigates RM over-optimization but not verifier exploits
