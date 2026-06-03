# SparseGPT / Wanda One-Shot Pruning

**One-liner:** Prune LLMs to 50%+ sparsity in one shot, no retraining — SparseGPT solves a layer-wise Hessian-based reconstruction per weight, Wanda uses a near-free weight×input-activation-norm importance score; both are the reference pruning baselines.

## The formula / definition

Both prune per-layer to match the dense layer's output `WX`, comparing within each output-row.

**Wanda** importance for weight `W_ij`:

```
S_ij = |W_ij| · ||X_j||_2
```

where `||X_j||_2` is the L2 norm of input feature `j` over the calibration batch. Per output row, drop the lowest-`S` weights. No weight updates, no inverse Hessian — just one calibration forward pass. Comparison is per-row (per-output), NOT global.

**SparseGPT** solves the layer-wise OBS objective `min ||WX - W'X||^2` using the local Hessian `H = XXᵀ + λI`. It prunes column-by-column; after dropping a weight it updates the remaining weights in that column block via the inverse-Hessian formula (OBS/OBQ), reusing a Cholesky of `H^{-1}` so cost is ~`O(d_hidden^3)` per layer, not per weight. Removed weight `w_q`, optimal update:

```
δ_remaining = -(w_q / [H^{-1}]_qq) · H^{-1}_{:,q}
```

Wanda ≈ SparseGPT with a *diagonal* Hessian approximation and no weight update.

## Where it appears

- **SparseGPT (Frantar & Alistarh 2023)** — first one-shot pruner that scales to 175B (OPT, BLOOM) at 50% sparsity with minor perplexity loss; reuses the GPTQ machinery (same inverse-Hessian solver).
- **Wanda (Sun et al. 2023)** — matches SparseGPT at a fraction of the cost (no solve); the cheap default baseline in pruning papers.
- **2:4 semi-structured sparsity** — both emit 2:4 masks for NVIDIA sparse Tensor Core (Ampere+) ~2x speedup; the practical deployment path since unstructured sparsity rarely accelerates dense kernels.

## Common mistake

Treating unstructured 50% sparsity as a 2x speedup. Unstructured masks give memory savings but little/no wall-clock gain on GPUs — you need structured (2:4) or actual sparse kernels. Also: pruning importance is computed *per output row*, not globally, and depends on calibration data (activation norms), so it interacts with activation outliers like quantization does.

## See also
- [[2-4-semi-structured-sparsity]] — the hardware-friendly mask both methods target for real speedup
- [[gptq]] — SparseGPT shares its inverse-Hessian OBS solver; pruning ↔ quantization duality
- [[structured-vs-unstructured-pruning]] — why the mask pattern, not the sparsity level, determines latency
