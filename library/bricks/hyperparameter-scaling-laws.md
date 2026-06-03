# Hyperparameter Scaling Laws (LR / batch / wd)

**One-liner:** Fitted power laws giving optimal learning rate, batch size, and (independent) weight decay as functions of model size N and tokens D, so HPs are *predicted* from a budget-driven fit on small proxies rather than swept at target scale.

## The formula / definition

Empirically, the loss surface over (η, B) at a fixed (N, D) is locally convex with a flat basin, and the basin's location moves smoothly with scale. Fit power laws:

η_opt ∝ N^a · D^b   (often a < 0: bigger models want *smaller* LR)
B_opt ∝ D^c          (optimal batch grows with data/compute, not model size)

**Step-law (2025):** η_opt ≈ C₁ · N^{-0.45} · D^{0.18}, B_opt ≈ C₂ · D^{0.55} — and crucially the loss landscape over (η, B) is *convex and broad*, so being near-optimal is cheap. Coefficients are architecture/data dependent.

**DeepSeek LLM (2024):** fit η_opt and B_opt against compute C = 6ND (not N alone): η_opt ∝ C^{-0.125}, B_opt ∝ C^{0.327}.

**Weight decay (the subtle one):** what matters is the *effective LR* set by the AdamW timescale. With decoupled WD λ and LR η, the steady-state weight norm and the equilibrium are governed by the product, with characteristic timescale τ ≈ 1/(η·λ). MiniCPM / WSD-era practice: scale λ so τ stays a fixed fraction of total steps, i.e. λ ∝ 1/(η · D). Don't hold λ constant while changing η.

## Where it appears

- **DeepSeek LLM (2024)** — fits η_opt, B_opt vs compute budget C to set HPs for a 67B run from small-scale sweeps; reports the optimal-allocation power laws above.
- **MiniCPM (2024)** — WSD schedule + scaling-law-predicted LR/batch; couples weight decay to the η·λ timescale rather than tuning it independently.
- **Step-law / "Predictable Scale" (2025)** — large-grid fit (thousands of runs) showing the (η, B) loss basin is convex and the optima are universal power laws transferable across N, D, and data mixtures.
- **μP / maximal-update parameterization** — orthogonal mechanism: μP transfers η *across width* via reparameterization; HP scaling laws fit η *across compute (N and D)*. Modern recipes use μP for width + a fitted η(D) for the data axis.

## Common mistake

Fitting η_opt as a function of N alone and assuming it's data-independent. Optimal LR also depends on D (token count / training length): for over-trained models the η that minimizes final loss is *lower* than the short-run optimum. Holding weight decay fixed while you change LR or training length is the related error — it silently changes the effective-LR equilibrium (η·λ timescale) and shifts the optimum.

## See also
- [[maximal-update-parameterization]] — the complementary mechanism: transfers LR across *width* by reparameterization rather than fitting a power law
- [[critical-batch-size-gradient-noise-scale]] — gives the theory for why B_opt grows with compute/data
- [[decoupled-weight-decay]] — why WD must be coupled to η (the η·λ effective-LR timescale)
