# Feature Learning vs Lazy Training

**One-liner:** A dichotomy in training dynamics — whether internal representations actually move (rich / feature-learning regime) or stay frozen at init while only the readout adapts (lazy / kernel regime); which regime you land in is set by parametrization and LR scaling, not by the architecture, and it's why real nets beat their NTK.

## The key insight

Linearize the network about init: f(θ) ≈ f(θ₀) + ∇_θf(θ₀)·(θ−θ₀). **Lazy** means this Taylor expansion stays accurate for the whole run — features ∇_θf don't change, so training ≡ kernel regression with the *fixed* NTK Θ₀. **Rich** means the relative weight movement is Θ(1), the Jacobian/NTK evolves, and hidden features adapt to the data.

The knob is the **output scale / parametrization**. Write f = α·g(θ). Chizat–Bartlett (2019): large α ⇒ lazy (a small Δθ already produces the needed output change, so weights barely move). The same effect comes from large width under the standard/NTK parametrization (1/√n init scaling).

**abc-parametrization (Yang–Hu Tensor Programs IV/V):** parametrize layer ℓ as weights init ~ n^{−2a_ℓ}, multiplier n^{−b_ℓ}, LR ~ n^{−c}. There is a *unique* stable scaling that keeps feature updates Θ(1) as width n→∞ — that is **μP (maximal update parametrization)**. Every other stable scaling collapses to a kernel limit (NTK regime). So the regime is a property of the scaling exponents, not a hyperparameter you tune by hand.

## The two regimes

- **Lazy / kernel:** ‖θ−θ₀‖ = o(1) (relative), NTK ≈ const, no feature learning. Induced by large width (NTK param), large output α, or tiny LR. Network = its own linearization; provably converges but generalizes like a kernel.
- **Rich / feature-learning:** Θ(1) weight movement, NTK evolves, representations specialize (e.g. neurons align to data, low-rank structure emerges). Reached via μP / mean-field (1/n scaling) or small init. This is where deep nets outperform their NTK.

## Where it appears

- **μP / Tensor Programs (Yang & Hu)** — μP is *defined* as the unique parametrization in the feature-learning regime; this is what makes HP transfer across width work (lazy limits would transfer trivially but not learn).
- **Mean-field theory of two-layer nets (Chizat–Bartlett, Mei–Montanari, Rotskoff–Vanden-Eijnden)** — the rich limit as a Wasserstein gradient flow over neurons.
- **Scaling / large-model practice** — frontier pretraining lives firmly in the rich regime; lazy training is the failure mode you engineer *away* from (μP, small init, large LR).
- **Grokking & emergence** — the delayed jump is the slow escape from a lazy-ish memorizing solution into learned features.

## Common mistake

Believing wider always = lazier and therefore worse. Width alone doesn't fix the regime — *parametrization* does. Under μP an infinitely wide net still learns features; under NTK/standard parametrization it goes lazy. So "infinite width ⇒ kernel" is a statement about a *choice of scaling*, not an inevitability of overparameterization.

## See also
- [[ntk]] — the fixed-kernel object that exactly describes the lazy regime
- [[maximal-update-parameterization]] — the unique scaling that stays in the feature-learning regime
- [[implicit-bias]] — small-init/rich training is where the low-norm/low-rank biases actually bite
