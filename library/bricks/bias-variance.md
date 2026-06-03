# Bias–Variance Tradeoff

**One-liner:** Expected test error decomposes as Bias² + Variance + irreducible Noise; the classical claim that capacity trades bias for variance (the U-shape) breaks in the overparameterized regime, where more capacity *lowers* variance (see double descent).

## The decomposition

For squared loss, expectation over training sets D, at a fixed point x with target y = f(x) + ε, Var(ε) = σ²:

E_D[(y − ĝ_D(x))²] = (f(x) − E_D[ĝ_D(x)])² + E_D[(ĝ_D(x) − E_D[ĝ_D(x)])²] + σ²
                   =        Bias²          +              Variance               + Noise

- **Bias²** — error from the model class being too rigid to represent f (underfitting).
- **Variance** — sensitivity of the fit to the particular training sample (overfitting).
- **Noise σ²** — irreducible; no model can beat it.

Clean additive split holds for squared loss. For 0–1 loss / cross-entropy the decomposition is not additive — analogues exist (Domingos 2000) but the geometry differs.

## The classical story vs. what actually happens

Classical: as capacity grows, Bias² falls and Variance rises → total error is U-shaped, optimal at the "sweet spot." This is the textbook regularization picture (ridge, early stopping, dropout all trade a bit of bias for less variance).

Modern (Belkin et al. 2019, "double descent"): push capacity *past* the interpolation threshold (params ≫ data, train error = 0) and test error falls *again*. In the overparameterized regime, the implicit bias of SGD selects a low-norm minimum-variance interpolator, so variance decreases with width. The U-shape is only the left half of the curve.

## Where it appears

- **Double descent** (Belkin 2019; Nakkiran 2020 "Deep Double Descent") — the headline failure of the naive U-shape; modern nets live to the right of the peak.
- **Random forests / bagging** — variance reduction by averaging decorrelated high-variance trees (bias roughly fixed).
- **Ensembling & deep ensembles** — averaging M independent models cuts the variance term ~1/M, leaving bias untouched; why ensembles reliably help.
- **Regularization at scale** — weight decay, label smoothing, early stopping all read as "spend bias to buy variance," but their effect inverts/weakens in the interpolating regime.

## Common mistake

Believing the U-shape is universal and that "bigger model ⇒ more variance ⇒ overfit." In the overparameterized regime that frontier models occupy, scaling capacity (and the implicit bias of the optimizer) *reduces* test error — the classical tradeoff does not describe deep nets past interpolation. The decomposition is still valid; the monotone "variance rises with capacity" intuition is what's wrong.

## See also
- [[double-descent]] — the phenomenon that breaks the classical U-shape
- [[implicit-bias]] — why overparameterized interpolators generalize (low variance, not high)
- [[scaling-laws]] — empirically, more capacity keeps lowering loss, contra the naive tradeoff
