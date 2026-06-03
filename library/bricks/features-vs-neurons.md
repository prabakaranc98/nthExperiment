# Features vs Neurons (Polysemanticity)

**One-liner:** The model's basis of computation (neurons / residual-stream dimensions) is not its basis of meaning (features); a single neuron is typically **polysemantic** — firing for many unrelated concepts — because the network packs more features than it has dimensions via superposition.

## The key insight

A **neuron** is a literal MLP/activation coordinate, a basis vector chosen by the architecture. A **feature** is an interpretable direction in activation space that corresponds to one human-meaningful concept. These coincide only by accident.

- **Polysemanticity:** one neuron's activation is a mixture of multiple feature signals. E.g. an InceptionV1/transformer neuron that lights up for "cat faces," "fronts of cars," AND "Hebrew text."
- **Why it happens:** if a layer needs to represent m features but has only n < m neurons, sparse features can be encoded as **near-orthogonal directions** in the n-dim space (superposition). Decoding requires the whole vector, not a single coordinate. Formally, feature activations f ∈ ℝ^m are recovered from activations x ∈ ℝ^n as f ≈ ReLU(W_dec^T x), with W_dec ∈ ℝ^{n×m} an overcomplete dictionary (the SAE objective).
- **Linear representation hypothesis** is the bet that features ARE (mostly) linear directions, so the right basis is a rotation/overcomplete dictionary away — recoverable, just not axis-aligned.

So: neurons are the wrong unit of analysis. Probing or steering on individual neurons gives garbled, entangled results; the goal of mech-interp is to find the feature basis.

## Where it appears

- **Olah et al., "Zoom In" / Circuits (2020)** — first systematic documentation of polysemantic neurons in vision models; motivated the search for features over neurons.
- **Elhage et al., "Toy Models of Superposition" (2022)** — shows polysemanticity is the *symptom* and superposition is the *cause*; derives when a model packs features vs. dedicates neurons (sparsity-dependent).
- **Sparse autoencoders (Anthropic "Towards Monosemanticity" 2023; "Scaling Monosemanticity" 2024; OpenAI/GDM SAE work)** — the standard tool to **disentangle** polysemantic activations into many monosemantic features; Claude 3 Sonnet SAE found millions of interpretable features.
- **Steering vectors / activation steering** — work because features are directions, but get contaminated when the direction overlaps a polysemantic neuron's other meanings.

## Common mistake

Treating a neuron's max-activating dataset examples as "what the neuron means," then declaring the model interpretable. A polysemantic neuron has *several* meanings; reading off only the top examples cherry-picks one mode and hides the rest. Also wrong: assuming features must be axis-aligned to neurons — the entire reason SAEs and dictionary learning exist is that they are not.

## See also
- [[superposition]] — the mechanism that forces neurons to be polysemantic
- [[sparse-autoencoders]] — the method that recovers monosemantic features from polysemantic activations
- [[linear-representation-hypothesis]] — the assumption that features are (recoverable) linear directions
