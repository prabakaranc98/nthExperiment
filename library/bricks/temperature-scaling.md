# Temperature Scaling

**One-liner:** A one-parameter post-hoc calibration that divides logits by a single learned scalar T before softmax — fixes over/under-confidence without touching accuracy (argmax is T-invariant); the cheapest, most-cited calibration baseline (Guo et al., 2017).

## The formula

Given logits z ∈ ℝ^K, replace softmax(z) with:

  p̂ = softmax(z / T),  T > 0

Fit T on a held-out validation set by minimizing NLL (cross-entropy) on logits:

  T* = argmin_T  − Σᵢ log softmax(zᵢ / T)[yᵢ]

- T > 1 → softens (reduces confidence; fixes overconfidence — the common case for modern nets)
- T < 1 → sharpens (increases confidence)
- T = 1 → no change
- T → ∞ → uniform; T → 0⁺ → one-hot

**Key property:** argmax_k (z_k / T) = argmax_k z_k for any T > 0. Accuracy, top-1, and ranking are unchanged; only the *probabilities* move. Measured via ECE / reliability diagrams, not accuracy.

## Where it appears

- **Guo et al. "On Calibration of Modern Neural Networks" (2017)** — origin; showed deep nets are systematically overconfident and that single-T temperature scaling beats Platt scaling, isotonic regression, and matrix/vector scaling on ECE.
- **Knowledge distillation (Hinton, 2015)** — same z/T softening on the *teacher* to expose dark knowledge; here T is a fixed hyperparameter, not fit, and loss is scaled by T² to keep gradients balanced.
- **LLM decoding** — the `temperature` sampling knob is exactly z/T applied at generation (T=0 ⇒ greedy). Sampling temperature ≠ calibration temperature, though the operation is identical.
- **Selective prediction / OOD & hallucination gating** — calibrated p̂ thresholds for abstention; common preprocessing before conformal or semantic-entropy methods.

## Common mistake

Believing temperature scaling improves accuracy or that a calibrated model is more correct. It only rescales confidence — argmax is invariant. Two other traps: it cannot fix *miscalibration that varies across the input space* (it's a single global scalar — class-conditional or accuracy-vs-confidence interactions need richer methods), and under distribution shift the fitted T no longer holds, so it gives a false sense of calibration on OOD data.

## See also
- [[calibration]] — ECE, reliability diagrams, and proper scoring rules that temperature scaling targets
- [[knowledge-distillation]] — reuses the identical z/T softening to transfer teacher soft labels
- [[conformal]] — distribution-free coverage sets vs. point-probability recalibration
