# Faithfulness & Completeness of Explanations

**One-liner:** Two orthogonal axes for judging an interpretability claim — *faithful* = the explanation reflects the mechanism the model actually uses (causal, not plausible); *complete* = it accounts for the full behavior, not a cherry-picked slice — operationalized via ablation, activation patching, and necessity/sufficiency tests.

## The definitions

**Faithfulness (causal):** an attributed component C *causes* behavior B. Tested by intervention, not correlation. Two complementary directions:
- **Necessity:** ablate/corrupt C → behavior B degrades. (If you can remove it and nothing changes, it wasn't the mechanism.)
- **Sufficiency:** patch C from a clean run into a corrupted run → B is restored. (If restoring just C recovers B, C carries the relevant signal.)

The standard metric is the **patching effect** on a logit-difference metric:

`ΔM = M(corrupt run + patch C from clean) − M(corrupt run)`

normalized so 0 = no effect, 1 = full clean behavior recovered. High recovery = C is a sufficient locus.

**Completeness:** the union of identified components explains (nearly) all of the behavior — the recovered effects sum to ~100% of the clean-vs-corrupt gap, with no large unexplained residual. A circuit is *incomplete* if ablating everything you found still leaves the behavior partly intact (a backup/parallel path you missed).

## Where it appears

- **Activation patching / causal tracing** (ROME, IOI circuit, Wang et al. 2022) — the core faithfulness test; localizes which residual-stream sites and heads are necessary/sufficient.
- **Sparse autoencoders & circuit analysis** (Anthropic 2024, attribution graphs 2025) — features are validated by causal ablation; "completeness" = reconstruction error + how much of the loss the features recover.
- **CoT faithfulness** (Anthropic/Lanham 2023, "Reasoning Models Don't Always Say What They Think" 2025) — does the stated chain-of-thought reflect the actual answer-determining computation? Tested by perturbing/truncating the CoT and watching the answer.
- **Faithfulness metrics** (comprehensiveness & sufficiency, DeYoung ERASER 2020) — erase top-attributed tokens (comp.) vs. keep only them (suff.).

## Common mistake

Conflating **faithfulness with plausibility**. A saliency map or a fluent CoT can look convincing to a human (plausible) while being causally irrelevant — the model would output the same thing with that "reason" ablated. Plausibility is a human-judgment property; faithfulness requires an intervention. The second mistake: declaring a circuit found without checking completeness, so backup heads / parallel paths silently carry the behavior.

## See also
- [[activation-patching-causal-tracing]] — the primary intervention test for both necessity and sufficiency
- [[cot-faithfulness-monitorability]] — faithfulness applied to chain-of-thought as a safety signal
- [[circuits]] — the object whose faithfulness and completeness you are evaluating
