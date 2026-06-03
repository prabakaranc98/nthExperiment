# Probing Classifiers

**One-liner:** Train a simple (usually linear) classifier on frozen activations to test whether a property — syntax, truth, sentiment, board state — is linearly decodable from a given layer; the decades-spanning "is X represented here?" diagnostic.

## The method

Freeze the model. Extract activations h = f_≤ℓ(x) ∈ ℝ^d at layer ℓ. Train a probe g on a labeled dataset {(h_i, y_i)} with the base model's weights frozen:

  ŷ = g(h),  g linear: ŷ = softmax(Wh + b)

High probe accuracy ⇒ y is (linearly) decodable from h. Compare across layers ℓ to localize where a property emerges. Keep g low-capacity on purpose: a powerful probe can learn the task itself, so accuracy stops being evidence about the representation.

## Where it appears

- **BERT-era NLP** — Hewitt & Manning structural probe (2019): a learned linear map under which L2 distance between token vectors recovers parse-tree distance. Tenney et al. "BERT rediscovers the NLP pipeline."
- **Truth / world-state probes** — Othello-GPT (Li et al. 2023): linear probes recover board state from a model trained only on move sequences; Marks & Tegmark, Burns et al. (CCS) probe for truthfulness directions.
- **Interpretability + steering** — a probe direction w is often reused as a steering vector or as the readout for a discovered feature; underpins the linear representation hypothesis.

## Common mistake

Reading probe accuracy as evidence the model *uses* that information. Probing is correlational decodability, not causal use — a property can be linearly present yet ignored by downstream computation. For causal claims you need intervention (activation patching / steering), not a probe. Also: control with a *selectivity* baseline (Hewitt & Liang) — accuracy on random-label control tasks — or a high-capacity probe inflates apparent structure.

## See also
- [[linear-representation-hypothesis]] — the assumption that justifies linear probes
- [[activation-patching-causal-tracing]] — the causal complement that probing alone cannot provide
- [[steering-vectors-activation-steering]] — probe directions repurposed as intervention vectors
