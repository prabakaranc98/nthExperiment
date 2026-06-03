# Cross-Attention Resampler / Q-Former

**One-liner:** A small bank of learned query tokens that cross-attend to a large, variable-length set of encoder (usually vision) features and compress them into a fixed, small number of output tokens for the LLM — decoupling visual token count from input resolution.

## The mechanism

Let `Z ∈ R^(M×d)` be the variable-length encoder features (M patches, M can be large/variable), and `Q ∈ R^(K×d)` be K *learned* query embeddings (K fixed, e.g. 32 or 64 — parameters, not data).

Stack of blocks; each is cross-attention from queries onto features:

    Attn = softmax( (Q W_q)(Z W_k)^T / sqrt(d) ) (Z W_v)     # Q is query, Z is key/value
    Q ← Q + Attn ;  Q ← Q + FFN(LN(Q))                       # plus optional self-attn among queries

Output is always K tokens regardless of M. Those K tokens (after a projection) become the visual prefix the LLM consumes. Key property: **output length K is constant**, so a 256-patch or 4096-patch image both cost K LLM tokens.

## Where it appears

- **Flamingo (DeepMind, 2022)** — "Perceiver Resampler" turns variable per-frame ViT features into 64 fixed tokens, fed via gated cross-attention into a frozen LM.
- **BLIP-2 (2023)** — the "Q-Former": 32 queries bridge a frozen image encoder and frozen LLM; pretrained with ITC/ITM/captioning before LLM hookup.
- **Qwen-VL, IDEFICS, mPLUG-Owl, MiniGPT-4** — resampler/Q-Former as the connector to keep visual token budget bounded.
- **2024-2026 trend** — many top VLMs (LLaVA-style) *dropped* the resampler for a plain MLP projector + dynamic tiling (AnyRes), trading token economy for fidelity; resamplers persist where token budget dominates (long video, many frames).

## Common mistake

Thinking it just "pools" or downsamples features. The queries are *learned content-addressed probes* — each query learns to attend to a recurring semantic role across images; it is not a fixed average-pool or strided conv, and the K queries are model parameters, not derived from the input.

## See also
- [[vlm-connector-projector]] — the MLP-projector alternative; resampler vs. projector is the core connector design choice
- [[multi-head-attention]] — the cross-attention here is standard MHA with separate query vs. key/value sources
- [[anyres-dynamic-high-resolution-tiling]] — competing approach to handling variable/high resolution without compressing token count
