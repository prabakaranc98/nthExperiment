# GPTQ

**One-liner:** One-shot post-training weight quantization to 3-4 bits that quantizes each layer's weights column-by-column while using approximate second-order (Hessian) information to greedily update the not-yet-quantized columns and minimize layerwise output reconstruction error — the canonical INT4 weight-only baseline.

## The objective and update

Per linear layer with weight W and calibration inputs X, minimize layerwise reconstruction error:

argmin_Ŵ ‖WX − ŴX‖²₂

The relevant curvature is the Hessian H = 2·XXᵀ (input second moments, NOT the loss Hessian). GPTQ inherits OBQ's optimal-update rule but fixes a quantization order (left-to-right, all rows in parallel) so it can use one shared Cholesky of H⁻¹ for the whole layer.

Quantize column q, then update the remaining columns to compensate:

δ = (w_q − quant(w_q)) / [H⁻¹]_qq   (per-column error scaled by inverse-Hessian diagonal)
W_remaining ← W_remaining − δ · [H⁻¹]_{q,remaining}

Key tricks that made it scale to 175B in hours on one GPU: (1) arbitrary order ≈ fixed order at scale, (2) lazy batched updates (block of 128 columns), (3) Cholesky reformulation for numerical stability. Activations stay FP16; only weights are quantized (W4A16). Group-wise scales (e.g. group size 128) are standard.

## Where it appears

- **OPT/BLOOM/Llama INT4 serving** — original GPTQ paper (Frantar et al., ICLR 2023); the reference 4-bit/3-bit weight-only method
- **AutoGPTQ / GPTQModel, exllama, Marlin kernels** — the dominant open-weight INT4 packaging + fast W4A16 GEMM kernels in vLLM/TGI
- **SparseGPT** — same OBQ-derived Hessian framework reused for one-shot pruning, and joint prune+quantize

## Common mistake

Thinking the "second-order info" is the loss Hessian. GPTQ never touches gradients or the training loss — H = XXᵀ is the Hessian of the *layerwise* reconstruction objective, estimated from a small calibration set (often ~128 sequences). This is also why it can overfit calibration data and why it leaves activation outliers untouched (it's weight-only), motivating SmoothQuant/AWQ/rotation methods.

## See also
- [[awq]] — contemporaneous weight-only INT4 method; activation-aware scaling instead of Hessian column updates
- [[sparsegpt-wanda-one-shot-pruning]] — same OBQ/Hessian machinery applied to one-shot pruning
- [[hadamard-rotation-based-quantization]] — rotations remove outliers GPTQ alone can't, enabling lower bits and W4A4
