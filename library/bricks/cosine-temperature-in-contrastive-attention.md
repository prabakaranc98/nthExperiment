# Cosine / Temperature in Contrastive & Attention Logits

**One-liner:** A scalar divisor applied to dot-product logits before softmax/sigmoid — fixed 1/√d_k in attention to control variance, learned (often capped) in contrastive losses to control sharpness — distinct from softmax itself, which it merely feeds.

## The formula / definition

**Attention scaling:** scores = QK^T / √d_k, then softmax. If q,k have i.i.d. zero-mean unit-variance entries, ⟨q,k⟩ has variance d_k, so the dot product grows ~√d_k. Dividing by √d_k restores O(1) variance → keeps softmax out of its saturated (low-gradient) regime.

**Contrastive temperature τ:** logits = (z_i · z_j) / τ where z are L2-normalized (so z_i·z_j = cosine ∈ [−1,1]). Small τ → sharp distribution, heavily weights hard negatives; large τ → soft, near-uniform. In CLIP, t = exp(τ_learned) (a learned *logit scale*, init ≈ log(1/0.07) ≈ 2.66) and is **clamped at ≤ log(100)** to prevent runaway.

These are the *same operation* — divide logits by a scalar — but the scalar's source (analytic vs learned) and purpose (variance vs sharpness) differ.

## Where it appears

- **Scaled dot-product attention (Vaswani 2017)** — the canonical 1/√d_k; without it deep nets train unstably as d_k grows.
- **CLIP (Radford 2021)** — learned temperature t = exp(τ), clamped at 100, jointly optimized; controls the InfoNCE sharpness.
- **SimCLR / MoCo** — fixed τ ≈ 0.07–0.2 on cosine logits; one of the most sensitive hyperparameters for downstream linear-probe accuracy.
- **SigLIP (Zhai 2023)** — learns *both* a temperature and a bias on cosine logits inside a sigmoid (not softmax) loss.
- **QK-norm + learned softmax scale** — modern LLMs (e.g. Gemma, ViT-22B) normalize q,k then learn the attention temperature to tame logit growth.

## Common mistake

Conflating this with **temperature in softmax sampling at decode time**. The contrastive/attention temperature divides logits *inside* a trained objective (it changes gradients and the optimum); decode-time temperature only reshapes an already-trained distribution at inference. Also: forgetting that contrastive τ assumes **L2-normalized** embeddings — without normalization, τ no longer maps to cosine and entangles with embedding norm.

## See also
- [[infonce-contrastive-loss-with-temperature]] — the loss whose sharpness τ controls
- [[scaled-dot-product-attention]] — the 1/√d_k origin
- [[qk-normalization]] — pairs with a learned attention temperature to stabilize logits
