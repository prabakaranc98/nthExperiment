# Matrix Rank

**One-liner:** The number of linearly independent rows/columns of a matrix = its effective dimensionality = the count of nonzero singular values; the whole reason low-rank adaptation, latent KV compression, and bottleneck layers work.

## The definition

For A ∈ ℝ^{m×n}, rank(A) = dim of column space = dim of row space (they're equal) = number of nonzero singular values σᵢ in A = UΣVᵀ.

Equivalent characterizations:
- rank(A) = m − dim(null(Aᵀ)) = n − dim(null(A))  (rank–nullity)
- Smallest r such that A = BC with B ∈ ℝ^{m×r}, C ∈ ℝ^{r×n}  (factorization view — this is LoRA)
- rank(A) ≤ min(m, n); "full rank" means equality

**Eckart–Young:** the best rank-r approximation (in Frobenius or spectral norm) is the truncated SVD — keep the top r singular values: Aᵣ = Σ_{i=1}^{r} σᵢ uᵢ vᵢᵀ. Error = σ_{r+1} (spectral) or √(Σ_{i>r} σᵢ²) (Frobenius).

## Effective vs exact rank

Real matrices (weights, activations) are almost never exactly low-rank, but their singular values decay fast. "Effective rank" = how many σᵢ are non-negligible (e.g., stable rank ‖A‖_F² / ‖A‖₂² = Σσᵢ² / σ₁²). This is what matters in practice, not the algebraic rank.

## Where it appears

- **LoRA** — freeze W, learn ΔW = BA with rank r ≪ d, so the update lives in an r-dimensional subspace. Works because fine-tuning updates are empirically low intrinsic rank.
- **MLA (Multi-head Latent Attention, DeepSeek-V2/V3)** — compress K,V into a low-rank latent c, shrinking the KV cache by storing the bottleneck instead of full K,V.
- **Bottleneck / autoencoders / MoE down-projections** — a narrow middle layer of width r forces representations through a rank-r channel.
- **Attention itself** — softmax(QKᵀ)/√d with head dim d_h ≪ N caps the rank of the score logits at d_h (low-rank attention bottleneck).

## Common mistake

Confusing rank with the matrix's shape or its number of parameters. A 4096×4096 matrix can have rank 8. Conversely, adding a low-rank update BA does NOT make the result low-rank — W + BA is generally full rank; LoRA's economy is in the *parameters trained*, not the rank of the final weight.

## See also
- [[svd]] — rank = count of nonzero singular values; truncated SVD is the optimal low-rank approximation
- [[lora]] — low-rank weight updates, the direct application
- [[kv-cache]] — MLA uses low-rank latent compression to shrink it
