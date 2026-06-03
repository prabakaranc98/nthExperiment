# Scaling Laws

**One-liner:** Loss follows power laws in model size (N), dataset size (D), and compute (C = 6ND); Chinchilla says train a smaller model on more data; 2024+ labs over-train for inference efficiency.

## The empirical result (Kaplan et al., 2020)

L(N) ∝ N^{-αN}  (loss decreases as power law with parameters)
L(D) ∝ D^{-αD}  (loss decreases as power law with data)
L(C) ∝ C^{-αC}  (loss decreases as power law with compute)

Approximately: αN ≈ αD ≈ 0.076 (rough estimates; varies by architecture and domain).

**The Chinchilla correction (Hoffmann et al., 2022):** for a fixed compute budget C, the optimal allocation is N* ∝ C^{0.5}, D* ∝ C^{0.5}. Kaplan et al. under-estimated the importance of data — you should scale N and D roughly equally.

## The inference-optimal insight (2024+)

Chinchilla-optimal minimizes training compute. But if you're going to serve the model billions of times, you want a *smaller* model trained on *more* tokens — the inference cost dominates. Most 2024 frontier models (Llama 3, Phi-4, Mistral) deliberately over-train relative to Chinchilla-optimal.

## The compute budget decomposition

C ≈ 6ND (forward + backward passes, approximate)

Where:
- N = number of non-embedding parameters
- D = number of training tokens
- 6 = ~2 for forward, ~4 for backward (approximate; varies)

## Where it appears

- Every pretraining decision at a frontier lab — how many parameters, how many tokens
- μP (Tensor Programs) — hyperparameter transfer relies on scaling being predictable
- Inference cost modeling — deciding between a 7B and 70B model for a deployment

## Common mistake

Treating the power law exponents as fixed. They vary by: domain, architecture, tokenization, data quality. The Chinchilla numbers are empirical estimates for transformer language models on web text. They don't directly transfer to code, math, proteins, or other domains without re-estimation.

## See also
- [[ntk]] — scaling laws have a theoretical account in the feature-learning regime
- [[edge-of-stability]] — training dynamics at scale connect to sharpness
