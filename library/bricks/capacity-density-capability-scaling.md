# Capacity-Density & Capability Scaling

**One-liner:** Beyond loss-vs-scale, newer fits relate *downstream capability* to compute, and "capacity density" measures effective capability per parameter — which is rising exponentially over time (roughly doubling every ~3-4 months in 2023-2025), meaning a model of fixed quality needs ever fewer parameters.

## The formula / definition

**Capability scaling (not just CE):** map loss to a benchmark via a saturating link. A common form (Gadre et al. 2024, "language models scale reliably"):
  Error(C) ≈ ε∞ + (loss(C) − L∞)^γ, or  Acc = σ(a·log C + b)
i.e. fit a sigmoid/exponential of the *log-compute* (or of reducible loss) rather than predicting the metric directly from C. The two-step recipe is: (1) predict loss L(N,D) via Chinchilla-style fit, (2) predict downstream metric M = f(L) with a monotone link.

**Capacity density (Densing Law, Xiao et al. 2024):** define the *effective parameter size* ρ of model M as the number of params an optimally-trained reference model would need to match M's downstream performance:
  density(M) = ρ_effective(M) / N_actual(M)
Empirical finding: max density across released models grows exponentially in time,
  ln(density_max) ≈ A·t + B,  doubling roughly every ~3.3 months.
Corollary: the params needed for a fixed capability halve on a similar cadence (an "inference-side Moore's law").

## Where it appears

- **Densing Law of LLMs (Xiao et al., 2024)** — defines capacity density, fits the exponential-in-time trend across open models (Llama, Qwen, MiniCPM, etc.).
- **"Language models scale reliably with over-training and on downstream tasks" (Gadre et al., 2024)** — predicts top-1 error from a scaling-law loss fit via a power-law link, extrapolating ~1000x.
- **Frontier release framing (2024-2026)** — Phi/MiniCPM/Qwen-small justify "small model, big capability" via density; distillation + data quality are the levers that move density, not just C = 6ND.
- **Emergent-ability debates** — capability-vs-scale curves underlie the "emergence is a metric artifact" critique.

## Common mistake

Conflating density gains with the fixed Chinchilla exponents. Capacity density rises because of *better data, distillation, architecture, and post-training* — it is an over-time trend across the model frontier, not a single training-run power law. Also: a smooth fit on a *continuous* metric (reducible loss, Brier) can look "emergent" only when you switch to a *thresholded* metric (exact-match accuracy) — choose the link function deliberately.

## See also
- [[scaling-laws]] — the loss-vs-N,D,C base that capability/density fits sit on top of
- [[emergent-abilities-the-mirage-critique]] — why downstream-capability curves can look discontinuous
- [[inference-optimal-over-training]] — over-training small models is how density gains get cashed out at serving time
