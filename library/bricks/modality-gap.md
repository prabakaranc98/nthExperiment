# Modality Gap

**One-liner:** In jointly trained contrastive spaces (CLIP-style), image and text embeddings collapse into two narrow, well-separated cones on the unit sphere rather than interleaving — a measurable geometric offset created at initialization and locked in by the temperature-scaled contrastive loss.

## The phenomenon (Liang et al., NeurIPS 2022)

Embeddings are L2-normalized to the unit hypersphere, yet each modality clusters in its own region. Measure the gap as the distance between modality centroids:

Δgap = ‖ (1/n) Σᵢ xᵢ − (1/m) Σⱼ yⱼ ‖₂,   with xᵢ, yⱼ on the sphere

Two causes:
1. **Cone effect** — deep nets with random init map any input into a narrow cone; the cone direction differs per encoder (per random seed), so the two towers start far apart.
2. **Contrastive loss preserves it** — InfoNCE with low temperature τ only needs paired (x,y) closer than negatives, not coincident. A constant gap can *lower* loss, so optimization keeps the cones separate.

Crucially, paired samples still rank-align: matching pairs have higher cosine than non-matching, even though no image embedding ever lands near its caption in absolute terms.

## Where it appears

- **CLIP / SigLIP** — the gap is universal across contrastive VLMs; embeddings of an image and its true caption sit ~0.8 cosine apart, not ~0.
- **Cross-modal retrieval / arithmetic** — naively doing text→image vector ops fails; shifting query embeddings by Δgap (or per-modality mean-centering) improves retrieval.
- **DALL·E 2 / unCLIP** — a learned "prior" network explicitly maps CLIP text embeddings into the CLIP image-embedding cone, bridging the gap before decoding.
- **Embedding-only captioning (CapDec, ZeroCap-style)** — train a decoder on text embeddings, add Gaussian noise / mean-shift to cross the gap at inference and decode images.
- **VLM connectors** — projecting frozen vision features into LLM token space contends with the same offset.

## Common mistake

Believing the gap means alignment failed or that closing it (e.g., forcing modality means to coincide) always helps. The gap can be benign or even useful — what matters is *relative* geometry (paired > unpaired cosine). Manipulating temperature or hard-collapsing the cones can hurt downstream zero-shot accuracy; the gap is a property of the loss landscape, not a bug to zero out.

## See also
- [[clip-contrastive-vision-language-pretraining]] — the training setup that produces the gap
- [[infonce-contrastive-loss-with-temperature]] — temperature τ controls how strongly the gap is preserved
- [[vlm-connector-projector]] — bridging image features into text space confronts the same offset
