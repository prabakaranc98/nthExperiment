# Self-Improvement / STaR Bootstrapping

**One-liner:** Iteratively sample reasoning traces from the current model, keep only those that reach the verified-correct answer (optionally rationalize failures by conditioning on the answer), then fine-tune on the survivors — bootstrapping a reasoner from a tiny seed with no human rationales.

## The loop

Given dataset {(xᵢ, yᵢ)} (problem, gold answer), one STaR iteration:

1. **Generate:** sample CoT rationale + answer: (rᵢ, ŷᵢ) ~ pθ(· | xᵢ)
2. **Filter (outcome reward):** keep (xᵢ, rᵢ) iff ŷᵢ = yᵢ
3. **Rationalize (failures only):** for wrong ones, sample r given the *answer in the prompt* — pθ(r | xᵢ, yᵢ) — keep if it now reaches yᵢ. Strip the answer hint from the saved prompt.
4. **Fine-tune** θ on the union of correct + rationalized traces.
5. Repeat. STaR re-trains from the **original base** each round (not the previous checkpoint) to avoid overfitting.

This is expectation-maximization: a hard-EM / policy-gradient surrogate where the latent is the rationale and the filter approximates ∑_r 1[ans(r)=y] · ∇log pθ(r|x).

## Where it appears

- **STaR** (Zelikman 2022) — the original generate-filter-finetune loop with rationalization for GSM8K/CommonsenseQA.
- **ReST / ReST^EM** (Gulcehre, Singh 2023–24) — formalize as Grow (sample) + Improve (filter by reward, reward-weighted/offline RL fine-tune) iterations; multiple Improve steps per Grow batch.
- **V-STaR** (Hosseini 2024) — also train a **verifier** (DPO) on correct *and incorrect* traces; use it for best-of-N at test time, fixing STaR's habit of discarding negatives.
- **RFT / rejection-sampling fine-tuning** and modern **RLVR** pipelines (R1-style, expert-iteration) are the same idea: it's the SFT-on-verified-rollouts limit of policy-gradient RL.

## Common mistake

Treating filtered SFT as free lunch and ignoring that it only reinforces what the model can *already* occasionally do — coverage is capped by pass@k of the seed model, so it stalls on problems never solved once. Also: rationalization leaks the answer, so forgetting to remove the answer hint from the training prompt teaches a shortcut, not reasoning; and re-using the previous round's checkpoint instead of the base model amplifies mode collapse.

## See also
- [[rejection-sampling-best-of-n]] — the per-iteration sampling+filtering primitive STaR loops on
- [[rlvr]] — STaR is the SFT/expert-iteration limit of RL with verifiable rewards
- [[model-collapse-curse-of-recursion]] — the failure mode when self-generated data lacks diversity/verification
