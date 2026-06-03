# Neural Tangent Kernel (NTK)

**One-liner:** In the infinite-width limit with the right parametrization, the network stays near its initialization and training becomes kernel regression with a fixed kernel (the "lazy" regime); the breakdown of this picture is where real feature learning — and most of scaling — lives.

## The formula / definition

For a network f(x; θ) the empirical NTK is the Gram matrix of parameter-gradients:

  Θ(x, x′) = ⟨∇_θ f(x; θ), ∇_θ f(x′; θ)⟩

Under gradient flow on MSE, the function evolves linearly in this kernel:

  df(x)/dt = −Θ(x, ·) · (f − y)

**Key limit (Jacot et al., 2018):** as width → ∞ under NTK parametrization (1/√width scaling), Θ converges to a *deterministic, constant-in-time* kernel Θ∞. Training is then equivalent to kernel ridgeless regression with Θ∞ — the parameters move only o(1), so the model is its own first-order Taylor expansion about init ("linearized network").

## The two regimes

- **Lazy / kernel regime:** weights barely move, Θ ≈ const, no feature learning. Induced by large width and large output scale α (Chizat & Bartlett "lazy training", 2019).
- **Feature-learning / rich regime:** Θ evolves, internal representations adapt. Reached via μP / mean-field parametrization (1/width scaling, not 1/√width). This is where deep nets beat their kernel.

## Where it appears

- **μP / Tensor Programs (Yang & Hu)** — explicitly chooses the parametrization that yields feature learning instead of the NTK regime; underpins hyperparameter transfer across width.
- **Wide-net theory & convergence proofs** — global convergence of GD on overparametrized nets via NTK positive-definiteness (Du, Allen-Zhu, Arora).
- **Scaling-law theory** — kernel/spectral arguments give power-law generalization exponents from the NTK eigenspectrum (Bordelon, Canatar, Bahri).
- **NNGP** — the forward-pass cousin: infinite width at init is a Gaussian process; NTK is its training-time analog.

## Common mistake

Treating NTK as a description of how real, trained deep networks behave. The lazy regime that makes NTK exact is the regime where networks *don't* learn features — finite-width SGD-trained nets are decidedly in the rich regime and outperform their NTK. NTK is the boundary you must leave, not the destination.

## See also
- [[scaling-laws]] — NTK spectrum gives a theoretical account of the power-law exponents
- [[double-descent]] — overparametrized kernel regression underlies the interpolation-threshold picture
- [[jacobian]] — the NTK is built from the network's parameter-Jacobian outer product
