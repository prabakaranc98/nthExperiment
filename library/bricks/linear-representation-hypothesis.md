# Linear Representation Hypothesis

**One-liner:** The conjecture that high-level concepts are encoded as linear directions in a model's activation (residual stream) space, so features superpose additively and concepts can be read, compared, and edited by vector arithmetic.

## The key claim

A concept `c` corresponds to a direction `v_c ∈ R^d`. Two operational readings:
- **Read (subspace/measurement):** the degree of `c` present in activation `h` is the projection `⟨h, v_c⟩` (after centering). A linear probe `σ(wᵀh + b)` recovers it.
- **Write (steering):** add/remove the concept by `h' = h + α·v_c`. The classic word2vec form: `v("king") − v("man") + v("woman") ≈ v("queen")` — relations are (approximately) constant offset vectors.

Park, Choe, Veitch (2024) formalize the duality: a **measurement** direction (what a probe reads) and an **embedding/intervention** direction (what steering moves) are generally *not the same vector* — they are dual under a causal inner product `⟨x,y⟩_C = xᵀ Σ⁻¹ y`, where Σ is the covariance of unembedding rows. Under that metric, unrelated concepts become orthogonal and addition behaves as expected.

## Where it appears

- **Steering vectors / activation addition (ITI, CAA, RepE)** — derive `v_c` from a mean difference of contrastive activations, add at inference to push behavior (truthfulness, sentiment, refusal).
- **Sparse autoencoders (Anthropic "Towards Monosemanticity", 2023–24)** — dictionary learning assumes features are linear directions in superposition; the decoder columns *are* candidate concept directions.
- **Refusal in LLMs (Arditi et al., 2024)** — refusal is mediated by a single direction; ablating it jailbreaks; adding it induces refusal.
- **Logit lens / linear probing / task vectors** — all presuppose that what you want lives in a readable linear subspace.

## Common mistake

Assuming *every* concept is linear and *globally* so. Counterexamples exist: some features are encoded **multi-dimensionally** (e.g. circular representations of days/months, Engels et al. 2024) or only become linear under the right (causal/whitening) inner product. "Linear direction" without specifying the metric — and whether you mean the probe direction or the steering direction — is the trap. Probe accuracy alone does not prove a feature is *used* causally.

## See also
- [[superposition]] — why many linear features pack into d dimensions, forcing non-orthogonality
- [[steering-vectors-activation-steering]] — the write-side application of the hypothesis
- [[probing-classifiers]] — the read-side test, and its correlation-vs-causation caveats
