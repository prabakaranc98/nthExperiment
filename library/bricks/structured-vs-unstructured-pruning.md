# Structured vs Unstructured Pruning

**One-liner:** Remove weights in hardware-friendly patterns (heads, channels, layers, 2:4 blocks) for real wall-clock speedup, or arbitrarily by importance (magnitude / Wanda / SparseGPT) for higher achievable sparsity that needs sparse kernels to pay off.

## The taxonomy

**Unstructured:** zero out individual weights wherever they're least important. Mask M ∈ {0,1}^(d×d), keep W ⊙ M. Reaches 50-90% sparsity at minimal quality loss, but a dense GEMM on a masked dense matrix is *not faster* — you need real sparse kernels (irregular memory access, hard on GPUs) to realize speedup.

**Structured:** remove whole units — attention heads, FFN neurons/channels, layers, or entire rows/columns. The remaining tensor is *dense and smaller*, so it runs faster on stock hardware immediately. Costs more quality per param removed because the constraint is coarse.

**Semi-structured (2:4):** the middle ground. In every contiguous block of 4 weights, exactly 2 are zero. NVIDIA Ampere+ Sparse Tensor Cores execute this as a ~2x-faster dense op via a compressed format + index metadata. Fixed 50% sparsity, real speedup, modest quality hit.

## Scoring: what to remove

- **Magnitude:** prune smallest |W|. Simple, weak for LLMs (ignores activations).
- **Wanda** (Sun et al., 2023): score = |W_ij| · ‖X_j‖₂ — weight magnitude times input-activation norm of that column. No retraining, no backprop, one forward pass of calibration data.
- **SparseGPT** (Frantar & Alistarh, 2023): one-shot, layerwise, solves a Hessian-based reconstruction (OBS-style) to pick + update remaining weights; handles 2:4 directly.

## Where it appears

- **2:4 on LLMs** — SparseGPT / Wanda export 2:4 masks; deployed with TensorRT-LLM / cuSPARSELt for ~1.5-1.8x decode speedup on H100/A100.
- **Structured LLM compression** — LLM-Pruner, Sheared-LLaMA prune layers/heads/dims then continue-pretrain to recover; produces a smaller dense model you can serve anywhere.
- **Combined with quantization** — sparsity + 4-bit (e.g. SparseGPT + GPTQ) stacks compression; common in edge/inference stacks.

## Common mistake

Believing high unstructured sparsity = proportional speedup. A 90%-sparse dense tensor on a normal GPU runs at *dense* speed — the zeros still get multiplied. Without genuine sparse kernels or a structured/2:4 pattern the hardware understands, you saved memory at best and got zero latency win. Match the sparsity pattern to what the target hardware can actually exploit.

## See also
- [[2-4-semi-structured-sparsity]] — the 2:4 pattern and its Sparse Tensor Core path in depth
- [[sparsegpt-wanda-one-shot-pruning]] — the one-shot importance scoring methods used to choose masks
- [[quantization]] — the other main weight-compression axis, frequently stacked with pruning
