# Eigendecomposition

**One-liner:** A = QΛQ⁻¹ factors a square matrix into its invariant directions (eigenvectors, columns of Q) and how much it stretches each (eigenvalues, diagonal of Λ); the spectral lens behind PCA, Hessian curvature, and spectral methods.

## The formula / definition

For square A ∈ ℝⁿˣⁿ, an eigenvector v ≠ 0 and eigenvalue λ satisfy:

  Av = λv     (A acts as pure scaling along v)

If A has n linearly independent eigenvectors, stack them as columns of Q and the λᵢ on the diagonal of Λ:

  A = QΛQ⁻¹     ⇒     Aᵏ = QΛᵏQ⁻¹

**Symmetric case (the one you actually use):** if A = Aᵀ (covariances, Hessians, Gram/Laplacian matrices), the spectral theorem guarantees real eigenvalues and an *orthonormal* Q, so Q⁻¹ = Qᵀ:

  A = QΛQᵀ = Σᵢ λᵢ vᵢvᵢᵀ

Eigenvalues found via det(A − λI) = 0; in practice via QR iteration / Lanczos, never the characteristic polynomial.

## Where it appears

- **PCA / whitening** — eigendecompose the data covariance XᵀX; top eigenvectors are principal directions, eigenvalues are explained variance. Whitening rescales by Λ^{−1/2}.
- **Hessian / loss curvature** — eigenvalues of ∇²L are the curvatures; largest (sharpness) drives the 2/η edge-of-stability threshold, the spectrum governs conditioning and second-order optimizers (K-FAC, Shampoo, Sophia).
- **Spectral methods** — graph Laplacian eigenvectors for spectral clustering / positional encodings; PageRank is the dominant eigenvector of the transition matrix.
- **NTK / mean-field analysis** — the NTK's eigenspectrum sets which functions are learned fast vs slow.
- **Stability of dynamics** — spectral radius max|λᵢ| < 1 controls whether RNN states or power iterates contract or explode.

## Common mistake

Conflating eigendecomposition with SVD. Eigendecomposition needs a *square* matrix and can have complex λ and non-orthogonal (even non-existent) eigenbasis when A is non-normal or defective. SVD works for *any* matrix with real non-negative singular values. They coincide only for symmetric PSD matrices, where singular values = eigenvalues. For a general A, the SVD of A relates to the *eigendecomposition of AᵀA*, not of A itself.

## See also
- [[svd]] — the rectangular, always-real generalization; equals eigendecomposition for symmetric PSD matrices
- [[edge-of-stability]] — the top Hessian eigenvalue hitting 2/η is the whole story
- [[matrix-rank]] — rank = number of nonzero eigenvalues (for diagonalizable A)
