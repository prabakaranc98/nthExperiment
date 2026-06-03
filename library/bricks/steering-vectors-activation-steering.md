# Steering Vectors / Activation Steering

**One-liner:** Add a fixed direction v to the residual stream at inference — h ← h + αv — to causally push generation toward (or away from) a concept, where v comes from contrastive activation differences, mean-difference (CAA), or a single SAE feature decoder vector.

## The formula / definition

Extract a direction at layer ℓ, then add it during the forward pass:

  h_ℓ ← h_ℓ + α·v̂,  v̂ = v / ‖v‖

**Contrastive / CAA (Rimsky et al., 2024):** average the difference of activations over paired prompts that differ only in the target concept:

  v = mean_{i} [ a_ℓ(x_i^+) − a_ℓ(x_i^−) ]

(e.g., x⁺ = "answer sycophantically", x⁻ = "answer honestly"; or A/B multiple-choice pairs read at the answer token).

**Refusal direction (Arditi et al., 2024):** v = mean harmful-prompt activation − mean harmless-prompt activation. *Ablating* it (projecting it out, h ← h − (h·v̂)v̂) jailbreaks; *adding* it forces refusal. Refusal is largely a single direction.

**SAE feature steering:** set v = decoder column W_dec[:, j] of feature j, clamp its activation. Golden Gate Claude (Anthropic, 2024) pinned one SAE feature high.

α is swept; too large breaks coherence. Steering is typically applied at one mid-to-late layer (or a band of layers).

## Where it appears

- **CAA / sycophancy & persona control** — mean-difference vectors at one layer steer refusal, sycophancy, corrigibility, myopia (Rimsky et al.).
- **Refusal-direction ablation** — Arditi et al. 2024: white-box jailbreak by removing one direction; the inverse of safety training.
- **Mechanistic interpretability** — SAE features as steering handles (Golden Gate Claude); ITI (Inference-Time Intervention) steers truthfulness along probe directions.
- **Concept editing / unlearning** — task-vector-style edits applied to activations rather than weights.

## Common mistake

Treating the steering vector as a *labeled* feature direction without testing it causally. A direction that linearly *decodes* a concept (a probe) is not necessarily the one that *controls* it — and adding v at inference is correlational evidence; you must also check ablation (does removing it suppress the behavior?) and off-target effects on capability.

## See also
- [[linear-representation-hypothesis]] — why a single direction can encode a concept
- [[sparse-autoencoders]] — SAE feature decoder vectors as a source of steering directions
- [[task-arithmetic-task-vectors]] — the weight-space analogue of adding a concept direction
