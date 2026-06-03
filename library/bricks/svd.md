# SVD & Low-Rank Approximation

**One-liner:** Every matrix factors as M = UΣVᵀ (orthonormal U, V; nonnegative diagonal Σ); truncating to the top-k singular values gives the *provably optimal* rank-k approximation (Eckart–Young), the foundation of LoRA, compression, and PCA.

## The factorization

For M ∈ ℝ^{m×n} of rank r:

M = UΣVᵀ = Σᵢ σᵢ uᵢ vᵢᵀ,  σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0

- U ∈ ℝ^{m×m}, V ∈ ℝ^{n×n} orthonormal (left/right singular vectors)
- Σ diagonal of singular values σᵢ = √(eigenvalues of MᵀM)
- σᵢ² are eigenvalues of MMᵀ and MᵀM; for symmetric PSD M, SVD = eigendecomposition

## Eckart–Young–Mirsky (the key theorem)

The best rank-k approximation under any unitarily-invariant norm is the truncation Mₖ = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ:

min_{rank(B)≤k} ‖M − B‖ = ‖M − Mₖ‖,  ‖M − Mₖ‖₂ = σₖ₊₁,  ‖M − Mₖ‖_F = √(Σ_{i>k} σᵢ²)

You cannot do better than zeroing the smallest singular values. Spectrum decay (how fast σᵢ → 0) determines how compressible M is.

## Where it appears

- **LoRA / DoRA** — adapt a weight as W + BA where BA is rank-r; the *premise* is that the update lies near a low-rank subspace. SVD of full fine-tuning deltas empirically shows fast spectral decay.
- **Model compression** — factor large weight matrices W ≈ U_k Σ_k V_kᵀ to cut params/FLOPs; basis for many low-rank pruning and KV-cache (low-rank head) tricks.
- **PCA / whitening** — PCA = SVD of mean-centered data; top singular vectors are principal directions.
- **Spectral norm & init** — σ₁ = ‖M‖₂ controls Lipschitz constant; used in spectral normalization and μP-style init analysis.
- **Randomized SVD** (Halko 2011) — sketch then factor; how SVD is actually computed on huge matrices.

## Common mistake

Assuming a matrix *is* low-rank because LoRA works. The full weight W is high-rank; only the *update* ΔW is assumed near-low-rank. Also: truncated SVD is optimal for the matrix, not for downstream task loss — minimizing ‖M − Mₖ‖_F ≠ minimizing model error, and activation-aware methods (e.g. weighted SVD) beat naive truncation.

## See also
- [[matrix-rank]] — rank is the count of nonzero singular values; SVD computes it numerically
- [[eigendecomposition]] — SVD generalizes it to non-square / non-symmetric matrices
- [[lora]] — low-rank adaptation built directly on the Eckart–Young premise
