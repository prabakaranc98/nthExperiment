# VLM Connector / Projector (LLaVA-style)

**One-liner:** A small trainable module (linear/MLP or cross-attention resampler) that maps a frozen vision encoder's patch embeddings into the LLM's token-embedding space, turning an image into a sequence of "soft" input tokens — the dominant recipe for cheap, capable VLMs.

## The definition

Vision encoder (e.g. CLIP/SigLIP ViT, frozen) produces patch features `V ∈ ℝ^{N×d_v}` (N patches). Connector `g_φ` maps them to LLM hidden dim `d_l`:

```
H = g_φ(V) ∈ ℝ^{M×d_l}          # M visual "soft tokens"
input = [emb(<sys>), H, emb(text_tokens)]   # concat into the LLM token stream
```

Two connector families:
- **MLP projector (LLaVA-1.5):** `g_φ(V) = W₂·GELU(W₁·V)`, applied per-patch → **M = N** (e.g. 576 tokens for 24×24 patches). No token reduction; cheap to train, scales token count with resolution.
- **Resampler / Q-Former / Perceiver (Flamingo, BLIP-2, Qwen-VL):** `M` learned queries cross-attend to `V` → fixed `M ≪ N` (e.g. 32–256), decoupling token budget from resolution at the cost of an attention bottleneck.

Training (LLaVA two-stage): **(1)** pretrain only `g_φ` on image-caption pairs (encoder + LLM frozen) to align modalities; **(2)** instruction-tune `g_φ` + LLM (often LoRA) on visual-instruction data. Loss is standard next-token CE on text only.

## Where it appears

- **LLaVA / LLaVA-1.5 / LLaVA-NeXT** — the canonical MLP-connector VLM; 1.5 swapped a linear layer for a 2-layer GELU MLP; NeXT adds AnyRes tiling so token count grows with resolution
- **BLIP-2 / InstructBLIP** — Q-Former resampler bridging frozen ViT and frozen LLM with learned queries
- **Flamingo / Idefics** — Perceiver Resampler + gated cross-attention layers *inside* the LM (not just at the input) — the alternative to in-context injection
- **Qwen2-VL, InternVL, Molmo, Pixtral** — production VLMs; mostly MLP-style connectors over SigLIP/CLIP with dynamic-resolution tiling; many now *unfreeze* the vision encoder in later stages

## Common mistake

Believing the connector "interprets" the image. It only **re-bases coordinates** — projecting vision features into the LLM's embedding geometry. Cross-modal *reasoning* happens in the LLM's self-attention over those soft tokens. Corollaries: (a) the visual token count, not pixels, drives prefill cost and KV-cache size — a resampler's fixed M can throw away fine detail (OCR/charts) that an MLP+AnyRes keeps; (b) a strong projector cannot rescue a weak/frozen vision encoder.

## See also
- [[clip-contrastive-vision-language-pretraining]] — the frozen encoder whose patch features the connector consumes (SigLIP is the 2024+ default)
- [[cross-attention-resampler-q-former]] — the resampler alternative to a per-patch MLP projector
- [[anyres-dynamic-high-resolution-tiling]] — how MLP-connector VLMs scale resolution by emitting more soft tokens
