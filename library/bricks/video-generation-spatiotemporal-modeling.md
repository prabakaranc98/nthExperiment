# Video Generation: Spatiotemporal Modeling

**One-liner:** Generate video by diffusing/denoising over spacetime latent patches from a causal 3D tokenizer, modeled with a DiT over factorized or full 3D attention — the central tension is joint spatial+temporal coherence vs. quadratic cost in frames × resolution.

## The pipeline / definition

1. **Causal video tokenizer (3D VAE):** encode RGB video x ∈ ℝ^{T×H×W×3} → latent z ∈ ℝ^{t×h×w×c} with compression (e.g. 4×8×8 or 8×8×8). *Causal* = the first frame encodes alone, so a single image is a length-1 video → joint image+video training. Temporal conv/attention only looks backward.
2. **Patchify spacetime:** flatten z into N = t·h·w spacetime patches ("spacetime latent patches", Sora). N scales with both duration and resolution → the sequence-length blowup that dominates cost.
3. **Diffusion/flow transformer (DiT):** predict noise/velocity εθ(z_τ, τ, c) with diffusion or flow-matching objective; condition c = text embedding (T5/CLIP) via cross-attention or adaLN, plus optional first-frame / pose / camera.
4. **Decode** denoised ẑ₀ back through the 3D-VAE decoder.

**Attention factorization** (the key cost lever): full 3D attention is O(N²) = O((t·h·w)²). Factorized spatial-then-temporal alternates blocks attending within a frame (cost O(t·(hw)²)) and across time at each spatial location (O(hw·t²)) — cheaper but weaker at fast motion. Modern systems (Sora, Veo) trend toward full 3D attention + heavy sequence/context parallelism.

## Where it appears

- **Sora (OpenAI, 2024)** — DiT on spacetime latent patches; variable duration/resolution/aspect by packing variable patch counts; recaptioning for dense text conditioning.
- **Movie Gen (Meta, 2024)** — flow-matching DiT, full bidirectional attention, factorized learnable spatial+temporal positional embeddings.
- **Veo / Veo 3 (Google DeepMind, 2024-25)** — latent diffusion video with joint audio generation.
- **Kling, CogVideoX, Wan, HunyuanVideo (2024-25)** — 3D causal VAE + DiT; CogVideoX uses expert-adaLN and 3D full attention.
- **Diffusion forcing / CausVid** — per-frame independent noise levels enable autoregressive long-rollout and real-time streaming generation.

## Common mistake

Treating temporal modeling as a cheap add-on to an image model (e.g. inflating a 2D UNet with 1D temporal layers and freezing the rest). This gives flicker and "morphing" because spatial and temporal features are entangled — the tokenizer's temporal compression and joint spacetime attention, not bolted-on temporal layers, are what produce real motion coherence. Also: factorized attention is an efficiency approximation, not equivalent to full 3D attention.

## See also
- [[diffusion-transformer]] — the DiT backbone that operates on the spacetime patches
- [[flow-matching]] — the training objective most modern video generators use
- [[diffusion-forcing-block-autoregressive-diffusion]] — per-frame noise schedule for long/streaming video
