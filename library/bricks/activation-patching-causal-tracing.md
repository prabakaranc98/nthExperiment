# Activation Patching / Causal Tracing

**One-liner:** A causal intervention that copies activations from a clean run into a corrupted run (or vice versa) at specific components, measuring each component's causal effect on the output; the workhorse localization tool of mechanistic interpretability.

## The procedure

Two forward passes on prompts that differ in the answer:
- **Clean run** x_clean (correct answer) — cache all activations.
- **Corrupted run** x_corr (wrong/noised answer) — the baseline.

Patch: rerun the corrupted prompt, but at component c (a layer, attention head, MLP, or single position in the residual stream) overwrite its activation with the cached clean value, then read the logit difference.

Effect metric (denoising / clean→corrupt direction):

  patching_effect(c) = [ M(x_corr | do(a_c ← a_c^clean)) − M(x_corr) ] / [ M(x_clean) − M(x_corr) ]

where M is usually the logit difference between correct and incorrect tokens. ~1 means c alone restores the answer; ~0 means c is causally irrelevant. The **noising** direction (corrupt→clean: patch corrupted activation into the clean run) tests necessity rather than sufficiency — they can disagree.

## Two corruption styles

- **Gaussian noising** (Meng et al., ROME) — add noise to subject-token embeddings.
- **Symmetric token swap** (Wang et al., IOI) — swap to a counterfactual prompt; cleaner because both runs are on-distribution. Prefer this; noising creates OOD activations and backup/self-repair behavior pollutes results.

## Where it appears

- **ROME / Causal Tracing (Meng et al., 2022)** — localized factual recall to mid-layer MLPs at the last subject token, motivating rank-one model editing.
- **IOI circuit (Wang et al., 2022)** — head-level patching reverse-engineered the indirect-object-identification circuit in GPT-2 small.
- **Attribution patching** — a linear (gradient-based) approximation that estimates all patching effects in two passes, used to scale to every component before exact verification.

## Common mistake

Reading a patching effect as "this component computes the answer." It only shows the component is causally *on the path* in this contrast — effects are relative to your specific corruption and metric, and self-repair (downstream heads compensating for an ablated one) routinely hides true importance under noising.

## See also
- [[attribution-patching]] — the cheap linear approximation that scales patching to all sites
- [[circuits]] — patching is how circuit hypotheses get causally validated
- [[residual-stream]] — the object you patch into and the basis for component decomposition
