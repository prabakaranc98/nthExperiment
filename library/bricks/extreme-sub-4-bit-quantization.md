# Extreme / Sub-4-Bit Quantization (2-bit, Ternary, BitNet)

**One-liner:** Push weights below 4 bits — 2-bit, ternary {−1,0,1} (BitNet b1.58 ≈ 1.58 bits), or binary — via learned scales, incoherence/rotation processing (QuIP#), additive vector codebooks (AQLM), or native quantization-aware training (BitNet); the compression-ratio frontier where naive round-to-nearest collapses entirely.

## The key insight

At ≤4 bits the per-element grid is so coarse that error is dominated by a few heavy-tailed weights and by *cross-weight correlations*, so the winning methods stop quantizing weights independently. Three families:

- **Incoherence processing (QuIP / QuIP#)** — multiply W by random orthogonal (Hadamard) rotations H_L W H_R so the spectrum is "incoherent" (no spiky entries, bounded ‖·‖_∞), then quantize the rotated matrix with a lattice codebook (E8). Rotation is undone in the matmul; spreads outliers across all dims so a uniform grid fits. Enables ~2-bit.
- **Additive codebook / vector quant (AQLM, QuIP# codebooks)** — represent a group of weights as a *sum of codewords* from learned codebooks: ŵ_group = Σ_m C_m[i_m]. Store only indices + small codebooks → <2.1 effective bits, decoded by table lookup. End-to-end fine-tuned to minimize layer output error.
- **Native QAT / ternary (BitNet, BitNet b1.58)** — train from scratch with a quantized forward pass. b1.58 ternarizes each weight via absmean scaling:

      γ = mean(|W|);   W_q = clamp(round(W / γ), −1, +1) ∈ {−1, 0, +1}

  STE passes gradients through round(); the master weights stay full precision. Result: matmuls become add/subtract (no multiplies), W1.58A8.

GPTQ-style Hessian error compensation (‖WX − ŴX‖²) still underpins many PTQ pipelines but is insufficient alone below ~3 bits.

## Where it appears

- **BitNet b1.58 / BitNet b1.58 2B4T (Microsoft)** — first trained-from-scratch ternary LLMs matching fp16 quality at 2B scale; bespoke bitnet.cpp kernels for CPU/edge.
- **QuIP#** — Hadamard incoherence + E8 lattice + fine-tuning; 2-bit Llama with usable perplexity.
- **AQLM** — additive quantization to ~2 bits, near-lossless on Llama-2/3 at 2-2.5 bits, the strongest extreme-PTQ baseline.
- **VPTQ, GPTVQ, OmniQuant, PV-Tuning** — vector-quant and learned-clipping refinements at 2-3 bits.

## Common mistake

Assuming 2-bit gives 8x speedup like it gives ~8x compression. These are weight-only, memory-bound wins (decode bandwidth); kernels still dequantize/lookup into fp16/bf16 for the matmul. The exception is *native* schemes like BitNet, where ternary weights + INT8 activations let the matmul become integer add/subtract — that is where real compute savings live. Also: PTQ to 2 bits without rotation, codebooks, or fine-tuning is essentially broken — RTN at 2 bits is unusable.

## See also
- [[quantization]] — the general affine/PTQ-vs-QAT framework these methods specialize
- [[hadamard-rotation-based-quantization]] — the incoherence trick QuIP# relies on
- [[quantization-aware-training]] — how BitNet trains through the ternary forward pass
