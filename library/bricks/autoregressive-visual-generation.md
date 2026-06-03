# Autoregressive Visual Generation (Next-Token / Next-Scale)

**One-liner:** Generate images/video/audio as sequences of discrete tokens with a causal transformer (`p(x) = Πᵢ p(xᵢ | x_{<i})`), either token-by-token in raster order or — in VAR — scale-by-scale (coarse-to-fine token maps), letting visual generation ride LLM infra (KV-cache, scaling laws, MoE).

## The two factorizations

**Next-token (raster AR):** tokenize image → VQ codebook indices flattened to a 1D sequence; predict left-to-right top-down. Cost: O(N) forward passes for N = H·W tokens (e.g. 256 for 16×16). Slow, breaks 2D locality, suffers a fixed raster order.

**Next-scale (VAR, Tian et al. 2024, NeurIPS best paper):** a multi-scale VQ tokenizer encodes an image as K token *maps* r₁,…,r_K at increasing resolutions (1×1, 2×2, …, h×w). Predict each whole map conditioned on all coarser maps:

```
p(r₁,…,r_K) = Π_{k=1}^{K} p(r_k | r_{<k})
```

All tokens *within* a scale are predicted in parallel (bidirectional attention inside a scale, block-causal across scales). Generation is O(K) ≈ O(log N) steps instead of O(N). Empirically VAR first beat diffusion (DiT) on ImageNet 256 FID and showed clean power-law scaling in params/compute.

## Where it appears

- **VAR / Infinity** — next-scale prediction; Infinity (2024) scales VAR to bitwise infinite-vocab tokenizers for high-res text-to-image.
- **LlamaGen, Parti, Chameleon, Emu3** — vanilla next-token raster AR over VQ tokens with a plain decoder transformer; Chameleon/Emu3 interleave image and text tokens for unified any-to-any models.
- **MAGVIT-v2 / VideoPoet / MagViT** — AR (or masked-AR) over spatiotemporal video tokens; lookup-free quantization (LFQ) for large vocabularies.
- **Audio** — AudioLM / MusicGen / VALL-E: AR over neural codec (RVQ) tokens.

## Common mistake

Conflating *next-scale AR* with *diffusion* or with *masked-parallel (MaskGIT)* generation. VAR is still pure autoregressive maximum-likelihood with teacher forcing and a KV-cache — there is no noising/denoising process and no iterative remasking; the "coarse-to-fine" structure comes from the multi-scale tokenizer, not from a noise schedule.

## See also
- [[vq-vae-discrete-visual-tokenizers]] — the discrete tokenizer (VQ/LFQ) that makes images a token sequence
- [[masked-parallel-token-generation]] — the non-AR (MaskGIT) alternative, often confused with next-scale
- [[diffusion-transformer]] — the continuous-latent competitor VAR/Infinity benchmark against
