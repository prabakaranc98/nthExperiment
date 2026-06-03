# Preference Data & Annotation Pipeline

**One-liner:** The end-to-end construction of comparison datasets — prompt sourcing, response sampling, pairwise/k-wise labeling, on- vs off-policy generation, and filtering — whose composition dominates final alignment quality far more than the choice of RLHF/DPO algorithm.

## The pipeline (and what each stage controls)

1. **Prompt sourcing** — distribution must match deployment (in-the-wild user logs > synthetic). Coverage of the prompt space sets the ceiling on what alignment can fix.
2. **Response sampling** — for prompt x, draw {y₁,...,yₖ} from a generator π_gen at temperature T. Diversity (T, top-p, multiple checkpoints) determines how informative the pairs are.
3. **Labeling** — annotator (human or LLM judge) produces a preference y_w ≻ y_l (pairwise) or a full ranking (k-wise). This is the data the reward model / DPO consumes.
4. **Filtering** — drop ties, low-agreement pairs, near-duplicate responses, length-confounded pairs, and prompts where both responses are bad.

**On-policy vs off-policy:** if π_gen = π (the model being aligned), pairs are *on-policy* — they cover the actual output manifold the policy will be optimized over. Off-policy pairs (from a different/older model) drift from where the policy lives, and the reward signal is uninformative or misleading there. This is the single biggest lever.

## The k-wise → pairwise conversion

A k-ranking decomposes into C(k,2) Bradley-Terry pairs; or fit a Plackett-Luce likelihood directly:
P(ranking) = ∏ᵢ exp(r(yᵢ)) / Σⱼ≥ᵢ exp(r(yⱼ)). InstructGPT collected k=4–9 rankings precisely because one annotation session yields many cheap pairs.

## Where it appears

- **InstructGPT / Llama 2 / Llama 3** — k-wise human rankings; Llama 2 explicitly used *iterative* on-policy collection (RLHF rounds re-sample from the current policy) and a 4-point margin to weight strong vs weak preferences.
- **Iterative / Online DPO** — regenerates pairs from the *current* policy each round to keep data on-policy, the main fix for offline DPO's distribution-shift failure.
- **RLAIF / Constitutional AI** — replaces human labelers with an LLM judge to scale annotation cheaply; quality now bounded by judge bias.
- **Rejection sampling / Best-of-N** — sample N, score, keep the top to build SFT or preference data (STaR, Llama 2 rejection-sampling stage).

## Common mistake

Obsessing over the optimizer (PPO vs DPO vs GRPO) while feeding it stale, off-policy, length-confounded pairs. Annotator agreement, prompt-distribution match, and on-policyness explain most of the variance in final quality; the loss function is a second-order knob. Length bias in particular silently turns your reward model into a "longer = better" detector.

## See also
- [[bradley-terry-model]] — the preference likelihood the labels are assumed to follow
- [[iterative-online-dpo]] — keeping the comparison data on-policy across rounds
- [[rlaif-constitutional-ai]] — swapping human annotators for an LLM judge to scale the pipeline
