# Superposition

**One-liner:** Networks represent more features than they have dimensions by encoding them as near-orthogonal directions in activation space, tolerating small interference noise to gain capacity — the core justification for sparse dictionary learning.

## The key insight

With d dimensions you get only d truly orthogonal directions, but the Johnson–Lindenstrauss lemma says you can pack exp(O(ε²d)) vectors with pairwise dot product ≤ ε. If features are **sparse** (active rarely, with probability p), a model can superimpose m ≫ d of them, since two features rarely fire together so the cross-talk interference is paid only occasionally.

Toy model (Anthropic 2022): input x → W (d×m, m features into d<m dims) → ReLU(WᵀWx + b) → x̂, train to reconstruct sparse x. As sparsity ↑, the learned columns Wᵢ go from clean orthogonal (one feature per neuron) to **overcomplete** with |Wᵢ·Wⱼ| > 0, often arranging into geometric structures (antipodal pairs, pentagons, tetrahedra). Interference cost ∝ (1−sparsity); capacity is "phase-transitioned" across features.

A feature is **monosemantic** if it gets a clean direction; **polysemantic** neurons are what you see because real features are superposed across the basis.

## Where it appears

- **Toy Models of Superposition** (Elhage et al., Anthropic 2022) — the foundational demonstration; introduces feature importance/sparsity phase diagram.
- **Sparse autoencoders / dictionary learning** (Towards Monosemanticity 2023; Gemma Scope, GPT-4 SAEs 2024) — SAEs are the inverse operation: recover the m overcomplete monosemantic features from the d-dim residual stream by enforcing sparsity.
- **Linear representation hypothesis** — superposition assumes features are linear directions, making steering vectors and probing well-posed.
- **Computation in superposition** (2024 work) — models not only store but *compute* over superposed features, complicating clean decomposition.

## Common mistake

Conflating superposition with mere polysemanticity. Polysemanticity is the observation (one neuron, many meanings); superposition is the mechanism and the claim that the true feature count exceeds the dimension. Also: assuming SAEs recover *the* ground-truth features — width, sparsity penalty, and feature splitting make the dictionary non-canonical (see SAE pathologies).

## See also
- [[sparse-autoencoders]] — the practical tool that un-mixes superposed features into a monosemantic dictionary
- [[features-vs-neurons]] — superposition is exactly why the feature basis ≠ the neuron basis
- [[linear-representation-hypothesis]] — the assumption that justifies treating features as packable directions
