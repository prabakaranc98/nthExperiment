# MXFP / Microscaling Block Formats (FP4/FP6/FP8)

**One-liner:** OCP-standardized low-precision formats where a small block of K elements shares one scale factor, so each value stores only a tiny FP mantissa/exponent while the block scale absorbs dynamic range — the substrate for Blackwell-era FP4/FP6/FP8 training and inference.

## The definition

A microscaling block is **K elements + 1 shared scale X**:

    value_i = X * P_i      for i = 1..K

- **K = 32** in the OCP MX spec (MXFP4, MXFP6, MXFP8, MXINT8).
- **X** is a shared **E8M0** scale: an 8-bit power-of-two (just an exponent, no mantissa), covering ~2^-127..2^127.
- **P_i** are the per-element private values in a tiny FP type:
  - **MXFP8**: E4M3 or E5M2 (8 bits)
  - **MXFP6**: E2M3 or E3M2 (6 bits)
  - **MXFP4**: E2M1 (4 bits — 1 sign, 2 exp, 1 mantissa; representable magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6})

Effective bits/element ≈ elem_bits + 8/K (e.g. MXFP4 ≈ 4.25 bits with K=32).

**NVFP4** (Blackwell, NVIDIA's variant): K=16 with a **two-level scale** — an FP8 (E4M3) per-block scale times an FP32 per-tensor scale. The E4M3 micro-scale is more accurate than MX's power-of-two E8M0, so NVFP4 generally beats MXFP4 in error at equal bits.

## Where it appears

- **OCP Microscaling Formats spec (2023)** — AMD/Arm/Intel/Meta/NVIDIA/Qualcomm standard defining MXFP4/6/8 and MXINT8; the interop contract.
- **NVIDIA Blackwell (B200/GB200, 2024-25)** — 5th-gen Tensor Cores natively MMA in MXFP8/MXFP6/MXFP4 and NVFP4; FP4 ~2x FP8 throughput. DeepSeek-style and NVIDIA recipes do FP8/FP4 GEMMs with block scales.
- **"Microscaling Data Formats for DL" (Rouhani et al., 2023)** — original MX training/inference study; MXFP6 near-lossless, MXFP4 needs care.
- **NVFP4 training (NVIDIA, 2025)** — 4-bit pretraining recipes using NVFP4 GEMMs with Hadamard/random-rotation pre-processing and stochastic rounding to hit BF16-comparable loss.
- **Inference quantization** — weights/activations in MXFP4/NVFP4 for KV and GEMM on Blackwell; competes with AWQ/GPTQ but is hardware-native.

## Common mistake

Treating "FP4" as a single number type. The block scale is the whole point: a lone E2M1 value has trivial range, but the **shared E8M0 (MX) or E4M3 (NVFP4) scale per K-element block** is what recovers dynamic range. Also: MXFP4 ≠ NVFP4 — different K (32 vs 16), different scale type (power-of-two vs FP8), and a per-tensor second level in NVFP4 — they are not bit-compatible and have different accuracy.

## See also
- [[fp8-low-precision-training]] — MXFP8 is the block-scaled successor; same per-tensor-scale problems it fixes
- [[extreme-sub-4-bit-quantization]] — MXFP4/NVFP4 are the hardware-native path to ~4-bit weights+activations
- [[hadamard-rotation-based-quantization]] — rotations spread outliers so 4-bit micro-blocks quantize cleanly
