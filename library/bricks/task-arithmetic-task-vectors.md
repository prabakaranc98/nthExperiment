# Task Arithmetic / Task Vectors

**One-liner:** A task vector is the weight delta τ = θ_ft − θ_base from fine-tuning; behaviors compose linearly in weight space (add to acquire, negate to forget), making it the algebraic backbone of training-free model merging like TIES and DARE.

## The formula / definition

Define the task vector for task t as the elementwise difference:

τ_t = θ_ft,t − θ_base

Then edit the base model by arithmetic on these vectors:

- **Learning / multi-task merge:** θ_new = θ_base + λ · Σ_t τ_t  (scaling λ ≈ 0.3–0.5 typical, tuned on val)
- **Forgetting / unlearning:** θ_new = θ_base − λ · τ_t  (negate to remove a capability, e.g. toxicity)
- **Analogy:** τ_C ≈ τ_A − τ_B + τ_D  ("A is to B as D is to C" in weight space)

All τ_t must come from the **same** θ_base (same pretrained checkpoint) for the geometry to hold.

## Where it appears

- **Ilharco et al. 2023 (Task Arithmetic)** — the original; showed addition/negation/analogy on CLIP and T5.
- **TIES-Merging (Yadav et al. 2023)** — resolves sign conflicts: TrIm small magnitudes → Elect dominant sign per param → merge only agreeing entries. Beats naive averaging by reducing destructive interference.
- **DARE (Yu et al. 2024)** — Drop And REscale: randomly zero ~90% of delta entries, rescale survivors by 1/(1−p). Sparsifies τ before merging; often stacked as DARE→TIES.
- **Model soups / WiSE-FT** — uniform averaging is the special case λ τ̄ with no sign handling.
- **LoRA merging** — adapter ΔW = BA is itself a low-rank task vector; same arithmetic applies to merge/serve multiple adapters.

## Common mistake

Assuming you can merge fine-tunes from **different base checkpoints** — task arithmetic requires a shared θ_base, since the vectors live in the same tangent space (loss-landscape mode connectivity). Also: naive *summing* without TIES/DARE causes destructive interference — overlapping parameters with opposite signs cancel, degrading every task. Merging is sign-resolution, not just averaging.

## See also
- [[weight-averaging-ema-model-soups]] — averaging fine-tunes is the no-arithmetic special case
- [[lora-merging-multi-adapter-serving]] — LoRA deltas are task vectors; same composition rules
- [[machine-unlearning]] — task-vector negation is a cheap (approximate) forgetting primitive
