# CLIP / Contrastive Vision-Language Pretraining

**One-liner:** Dual encoders (image tower + text tower) trained on web-scale (image, caption) pairs with a symmetric InfoNCE loss over in-batch positives/negatives and a learned temperature, yielding a shared embedding space that enables zero-shot classification via text prompts.

## The formula / definition

Encode batch of N pairs to L2-normalized embeddings: image `i_k`, text `t_k`. Logits = scaled cosine similarities, `logits = (I @ T.T) * exp(τ)` where `τ` is a learned scalar (parameterized in log-space, clamped, e.g. `exp(τ)<=100`). Symmetric cross-entropy over rows and columns with the diagonal as the positive:

```
labels = arange(N)
L_img  = CE(logits,   labels)   # each image -> its caption
L_txt  = CE(logits.T, labels)   # each caption -> its image
loss   = (L_img + L_txt) / 2
```

Each row is a softmax over N-1 in-batch negatives + 1 positive. Loss quality scales with batch size (more negatives); CLIP used 32k. Towers: ViT or ResNet for image; Transformer for text. Inference: zero-shot classify by embedding prompts ("a photo of a {class}") and taking argmax cosine similarity.

## Where it appears

- CLIP (Radford 2021) — original; foundation for zero-shot recognition and the canonical contrastive VLM recipe.
- SigLIP / SigLIP 2 — replaces the softmax InfoNCE with a per-pair sigmoid loss, removing the global all-gather normalization so it scales to smaller batches.
- VLM vision encoders — CLIP/SigLIP ViTs are the frozen (or lightly tuned) image backbone feeding the projector in LLaVA, Qwen-VL, PaliGemma, InternVL.
- Diffusion conditioning — CLIP text/image embeddings condition Stable Diffusion / unCLIP; also drives CLIP-score eval.

## Common mistake

Forgetting to L2-normalize before the dot product, or treating the temperature as a fixed hyperparameter. The cosine + learned temperature is load-bearing: it sets the softmax sharpness, and an unclamped `exp(τ)` blows up and destabilizes training.

## See also
- [[infonce-contrastive-loss-with-temperature]] — the exact loss CLIP instantiates
- [[siglip-sigmoid-contrastive-loss]] — the sigmoid alternative that drops batch-global normalization
- [[modality-gap]] — image and text embeddings occupy separate cones, not one blended manifold
