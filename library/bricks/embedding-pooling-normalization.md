# Embedding Pooling & Normalization

**One-liner:** Collapse a sequence of token embeddings into one fixed vector (mean / CLS / last-token / attention pool) then L2-normalize so that dot product equals cosine similarity — a tiny architectural choice that materially moves retrieval quality.

## The formula / definition

Given token embeddings H ∈ R^{T×d} from the encoder, pool to v ∈ R^d, then normalize:

- **Mean pool** (attention-masked): v = (Σ_t m_t · h_t) / (Σ_t m_t), where m_t is the attention mask (never average over pads)
- **CLS pool:** v = h_[CLS] (first token; the BERT convention)
- **Last-token pool:** v = h_{last non-pad} (the right choice for causal/decoder LMs, since only the last position attends to the full sequence)
- **Attention/weighted pool:** v = Σ_t softmax(w·h_t)_t · h_t (learned pooling head)

Then **L2-normalize:** v̂ = v / ‖v‖₂. Now ⟨û, v̂⟩ = cos(u, v) ∈ [−1, 1], and Euclidean distance ‖û − v̂‖² = 2 − 2·cos — so cosine, inner product, and L2 induce the same ranking on the unit sphere.

## Where it appears

- **Sentence-BERT / E5 / GTE / BGE** — mean pooling over the last layer + L2-norm is the dominant recipe for encoder-based dense retrievers; trained with InfoNCE so cosine is the scored quantity
- **Decoder embedders (e5-mistral, NV-Embed, gte-Qwen, GritLM)** — last-token pooling (often with an EOS/instruction-appended prompt) because causal masking means earlier positions can't see the whole input; NV-Embed swaps in a learned latent-attention pool
- **CLIP / SigLIP** — pool image and text towers then L2-normalize before the cosine/temperature-scaled contrastive loss
- **Matryoshka (MRL)** — normalize the full vector, then truncate-and-renormalize prefixes for adaptive dimensionality

## Common mistake

Mean-pooling **including padding tokens** (or pooling raw token embeddings before contextualization) — always apply the attention mask. The second classic error: using **last-token pooling on a bidirectional encoder** or **mean pooling on a causal decoder** — pooling must match the attention pattern. And: forgetting to L2-normalize, so the index does raw inner product where vector magnitude (a function of length/frequency, not semantics) silently dominates ranking.

## See also
- [[infonce-contrastive-loss-with-temperature]] — the training objective whose geometry assumes normalized embeddings on a sphere
- [[matryoshka-representation-learning]] — nested normalized prefixes for elastic embedding dimension
- [[dense-retrieval]] — the downstream system where the pooled, normalized vector is the index key
