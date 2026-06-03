# Vision Transformer (ViT) Patchification

**One-liner:** Split an image into a grid of non-overlapping P×P patches, flatten and linearly project each into a token embedding, add positional encodings, and feed the sequence to a standard transformer — the universal "image as a sequence of tokens" interface.

## The formula / definition

Image x ∈ ℝ^{H×W×C}, patch size P → N = HW/P² patches.

1. Reshape into patches x_p ∈ ℝ^{N×(P²·C)} (each patch flattened).
2. Linear projection (the "patch embedding"): z₀ⁱ = x_pⁱ · E,  E ∈ ℝ^{(P²·C)×D}.
3. Prepend learnable [CLS] token (ViT) or use mean-pool later.
4. Add positional embedding: z₀ = [x_cls; z₀¹; …; z₀ᴺ] + E_pos.
5. Feed z₀ to L standard transformer blocks (MHSA + MLP, pre-norm).

The projection in step 2 is **mathematically equivalent to a Conv2d with kernel=stride=P** — that is the standard implementation (`nn.Conv2d(C, D, kernel_size=P, stride=P)`). For ViT-B/16: P=16, 224×224 → N=196 tokens, D=768.

## Where it appears

- **ViT (Dosovitskiy et al., 2021)** — the original; showed a near-pure transformer beats CNNs given enough data (JFT-300M).
- **CLIP / SigLIP** — patchified ViT image encoder aligned to a text tower via contrastive loss.
- **VLMs (LLaVA, Qwen-VL, PaliGemma)** — ViT patch tokens projected through a connector into the LLM token stream; the connector is the modality bridge.
- **AnyRes / NaViT / Qwen2-VL** — variable-resolution patchification: tile or pack patches from native-resolution images instead of fixed 224² resizing.
- **MAE / DINOv2** — patch tokens are the unit of masking (MAE) or self-distillation (DINOv2).

## Common mistake

Treating patch size as a free hyperparameter without seeing the **sequence-length / compute tradeoff**: N scales as 1/P², and self-attention is O(N²). Halving P (16→8) quadruples token count and ~16×s attention FLOPs. Smaller patches give finer detail but explode cost — which is exactly why high-res VLMs need tiling (AnyRes) or token compression (resamplers) rather than just shrinking P.

## See also
- [[clip-contrastive-vision-language-pretraining]] — patchified ViT is the standard image encoder
- [[vlm-connector-projector]] — maps patch tokens into the LLM embedding space
- [[anyres-dynamic-high-resolution-tiling]] — escapes the fixed-grid patchification limit for high resolution
