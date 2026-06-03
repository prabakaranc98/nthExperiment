# Logit Lens

**One-liner:** Decode any intermediate residual-stream activation by projecting it through the final norm + unembedding to read the model's current best-guess token distribution at that layer — interpretability's cheapest "what is the model thinking now" probe.

## The formula / definition

For residual-stream state h_ℓ at layer ℓ (post-block, pre-final-norm), apply the model's *own* final head:

logits_ℓ = W_U · LayerNorm_f(h_ℓ)
p_ℓ = softmax(logits_ℓ)

where W_U is the unembedding (often tied to the embedding, W_U = W_E^T) and LayerNorm_f is the final pre-unembedding norm. No training — you reuse the existing readout on an earlier hidden state. Across layers you watch the predicted token "resolve": early layers garbage/positional, middle layers surface the answer, late layers sharpen it. Works because the residual stream is a single additive accumulator in one coordinate system (the read/write basis of W_U).

## Where it appears

- **nostalgebraist (2020)** — original blog post on GPT-2; coined "logit lens," showed the answer token often appears mid-stack.
- **Tuned Lens (Belrose et al., 2023)** — fits a per-layer affine probe A_ℓ h_ℓ + b_ℓ before W_U to correct the basis mismatch; far lower perplexity and bias than raw logit lens, and a more faithful causal trajectory.
- **Future Lens / DoLa** — DoLa (Chuang et al., 2024) contrasts early vs. late logit-lens distributions to reduce hallucination at decode time; multilingual work uses it to show models "think in English" internally.
- **Patchscopes (Ghandeharioun et al., 2024)** — generalizes the idea: instead of unembedding directly, patch h_ℓ into a fresh prompt and let the model verbalize it.

## Common mistake

Treating raw logit-lens distributions as a faithful readout for *every* model. The plain lens assumes h_ℓ lives in the same basis W_U expects; many models (and especially non-GPT-2 architectures) have a representational drift / norm mismatch that makes early-layer logit lens look like noise or systematically biased — that artifact is exactly why Tuned Lens exists. Absence of a clear prediction does not mean the feature is absent.

## See also
- [[residual-stream]] — the additive read/write channel the lens decodes
- [[softmax-bottleneck-logit-cap-final-layer-tying]] — tied W_U / final-norm geometry that makes the projection meaningful
- [[probing-classifiers]] — the trained-probe cousin (tuned lens is a constrained probe)
