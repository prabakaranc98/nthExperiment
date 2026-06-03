# AnyRes / Dynamic High-Resolution Tiling

**One-liner:** Feed a VLM high-resolution images by splitting them into a grid of native-resolution tiles plus one downsampled global thumbnail, encoding each independently with a fixed-input ViT, and concatenating all patch tokens so fine detail (text, charts, small objects) survives.

## The recipe

Given image of size H×W and a ViT that takes fixed s×s crops (e.g. 336 or 448):
1. Pick a tiling grid (a, b) from a candidate set (e.g. 1×1, 1×2, 2×2, 1×3, ... up to a max N tiles) that best matches the image aspect ratio / minimizes resize distortion.
2. Resize to (a·s)×(b·s), split into a·b non-overlapping tiles, encode each tile -> (s/p)² patch tokens (p = patch size).
3. Also encode a single global thumbnail resized to s×s for whole-image context.
4. Concatenate: tokens ≈ (a·b + 1)·(s/p)². Token count grows ~linearly with pixels, so a pixel-shuffle / token-merge downsampler (4 tokens -> 1) is usually applied to control sequence length.

Total LLM vision tokens with N tiles: **N_tokens = (N+1) · (s/p)² / r**, where r is the downsample factor.

## Where it appears

- **LLaVA-NeXT (1.6)** — originated "AnyRes": fixed grid of 336px CLIP crops + base image, tokens flattened with row separators.
- **InternVL 1.5+** — dynamic aspect-ratio tiling up to 40 tiles (448px), pixel-shuffle 4x token reduction; the dominant high-res-doc recipe.
- **Qwen-VL / Qwen2-VL** — Qwen2-VL drops fixed tiling for **naive dynamic resolution**: NaViT-style variable patch counts + 2D-RoPE, processing arbitrary resolutions without a fixed tile grid (the successor design).
- **MiniCPM-V, DeepSeek-VL2, Idefics2/3** — variations on slice-and-thumbnail for OCR/document/chart understanding.

## Common mistake

Forgetting that tiles are encoded **independently** by the ViT — there is no cross-tile attention inside the vision encoder, so an object spanning a tile boundary is split, and the model must re-stitch context via the LLM and the global thumbnail. Also: blindly cranking max-tiles explodes token count quadratically in resolution and dominates prefill cost, with diminishing accuracy gains.

## See also
- [[vision-transformer-patchification]] — the per-tile encoder turning each crop into patch tokens
- [[vlm-connector-projector]] — projects/merges the concatenated tile tokens into the LLM embedding space
- [[clip-contrastive-vision-language-pretraining]] — supplies the fixed-resolution ViT backbone each tile is run through
