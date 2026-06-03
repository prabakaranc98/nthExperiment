# Scalable Oversight

**One-liner:** The umbrella of techniques for supervising models on tasks where humans can't directly judge correctness — by amplifying, decomposing, or verifier-assisting human judgment so that supervision quality keeps pace with capability.

## The key insight

The core bet: **verification is easier than generation**, and **judging a decomposed sub-claim is easier than judging the whole**. Build a supervision signal H' from a weak human judge H plus model assistance M such that H' is more reliable than H alone, ideally without the human ever evaluating the full task.

Three canonical mechanisms:
- **Amplification (IDA):** H'ₜ₊₁ = Amplify(H, Mₜ); train Mₜ₊₁ to imitate/distill H'ₜ₊₁. Human + model team supervises the next model. Iterate.
- **Decomposition / Factored cognition:** break task T into subtasks {tᵢ}; human judges each tᵢ; recombine. Recursive Reward Modeling (RRM) trains a reward model on these human-judged sub-evaluations.
- **Debate:** two models argue opposite answers; human judges the transcript. Under optimal play, the equilibrium incentive favors true claims because lies are easier to refute (an asymmetry argument, not a guarantee).

Formal target — a **low-stakes** setting where we want supervision such that the trained policy's quality tracks the *true* objective rather than the *cheaply-checkable proxy* (avoiding reward hacking on the gap).

## Where it appears

- **AI Safety via Debate** (Irving et al., 2018; OpenAI 2024 prover-estimator debate) — judge-decided two-agent debate as the oversight signal.
- **Recursive Reward Modeling** (Leike et al., 2018) — RMs trained with model-assisted human eval, the backbone of RLHF on hard tasks.
- **Weak-to-strong generalization** (OpenAI, 2023) — empirical proxy: can a weak supervisor elicit strong-model capability? A measurable stand-in for the human→superhuman gap.
- **Constitutional AI / RLAIF** — model-generated critiques amplify thin human oversight into dense feedback.
- **Sandwiching experiments** (Bowman et al., 2022) — non-expert + model vs. expert ground truth, to test whether assistance closes the gap.
- **Process reward models** — supervising reasoning *steps* is decomposition applied to chains of thought.

## Common mistake

Treating scalable oversight as a solved alignment guarantee. It is an *empirical research program*, not a theorem — debate's truth-favoring property holds only under strong assumptions (honest equilibrium, bounded argument depth, a competent judge) and breaks under obfuscated arguments. It also assumes the human judge is at least *weakly* correct and not systematically exploitable; sycophancy and persuasive-but-wrong arguments attack exactly this.

## See also
- [[ai-safety-via-debate]] — the debate instantiation of scalable oversight
- [[weak-to-strong-generalization]] — the empirical proxy for the superhuman-supervision problem
- [[reward-hacking-over-optimization]] — what scalable oversight is meant to prevent
