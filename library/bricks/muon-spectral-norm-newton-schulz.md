# Muon / Spectral-Norm & Newton-Schulz Orthogonalization

**One-liner:** Replace a momentum-matrix gradient G with its nearest semi-orthogonal matrix UVᵀ (all singular values pushed to 1), computed cheaply by a few Newton-Schulz iterations in bf16 instead of an SVD — this is the core update primitive inside Muon.

## The formula

Given momentum buffer M (a 2D weight's gradient/momentum), the ideal update is the orthogonal polar factor of its SVD M = UΣVᵀ:

  O = UVᵀ = (M Mᵀ)^(−1/2) M = U·I·Vᵀ   (equalizes all singular values to 1)

SVD is too slow, so approximate via a quintic Newton-Schulz iteration on X (normalized so ‖X‖₂ ≤ 1):

  X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X

with tuned (a,b,c) ≈ (3.4445, −4.7750, 2.0315). ~5 iterations suffice. The cubic is the polynomial p(σ) acting on singular values, driving every σ → 1. Init: X₀ = M / ‖M‖_F (or /‖M‖₂). Then weight update W ← W − η·O, scaled by √(fan-out/fan-in) so the RMS norm matches AdamW's.

The deeper view: orthogonalization is a **steepest-descent step under the spectral norm** (not Euclidean). It caps the largest singular value of the update, controlling how much any input direction can be amplified.

## Where it appears

- **Muon optimizer** (Jordan et al., 2024) — orthogonalizes the Nesterov momentum of every 2D hidden weight; held the NanoGPT speedrun record
- **Frontier pretraining** — Kimi K2 / Moonshot scaled Muon to trillion-param MoE; many 2025 labs adopt it for hidden layers
- **Modular norm / spectral-norm theory** (Bernstein, Large) — Muon is the operator-norm-induced steepest descent; basis for "metrized deep learning"
- **Shampoo/SOAP lineage** — orthogonalization approximates a whitening/preconditioner step without storing full second-moment matrices

## Common mistake

Thinking Newton-Schulz here needs to *converge* to the exact polar factor like a numerical-precision routine. It deliberately does not — coefficients are tuned for fast, coarse convergence in low precision, and over-iterating or demanding high accuracy wastes compute. Also: it applies only to 2D matrices; embeddings, the LM head, scalars, and norm gains are kept on AdamW.

## See also
- [[muon-optimizer]] — the full optimizer that wraps this orthogonalization primitive
- [[shampoo-soap]] — related second-order preconditioning via matrix whitening
- [[svd]] — the exact factorization Newton-Schulz cheaply approximates
