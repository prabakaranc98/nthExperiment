# Softmax Bottleneck & Logit Cap / Final-Layer Tying

**One-liner:** A softmax over logits W·h is rank-limited by hidden dim d (the "softmax bottleneck"), and the output matrix W is often *tied* to the input embedding — both decisions interact with stability tricks like logit soft-capping and z-loss in modern LM heads.

## The key insight

Logits are `z = W h`, with `W ∈ R^{V×d}`, `h ∈ R^d`, then `p = softmax(z)`. The matrix of log-probabilities over all contexts has rank ≤ d (Yang et al., 2018, "Breaking the Softmax Bottleneck"). Since real next-token distributions are high-rank, a single softmax with `d ≪ V` *cannot* express them — capacity is capped by hidden size, not vocab size.

**Weight tying** sets `W = E` (the input embedding), saving `V·d` params and improving small-model PPL (Press & Wolf 2017; Inan et al. 2017). But tying couples input and output representation geometry, and the shared scale forces a careful interaction with the final LayerNorm/RMSNorm.

**Logit soft-cap** (Gemma 2): `z ← c · tanh(z / c)` with c≈30 (final) / 50 (attn), squashing extremes into (−c, c) to bound the softmax input and prevent runaway logits.

**z-loss** (PaLM): add `λ · (logsumexp(z))²` to the loss, pulling the log-partition toward 0 so logits don't drift large — cheaper, gradient-friendly alternative to capping.

## Where it appears

- **Mixture of Softmaxes (MoS)** — Yang et al.'s fix: `p = Σ_k π_k softmax(W_k h)`, K mixture components break the rank cap; rarely used now (compute cost) but the diagnosis is canonical.
- **Gemma 2 / Gemma 3** — tanh logit soft-capping on attention and final logits (Gemma 2; partly dropped for FlashAttention compatibility in later versions).
- **PaLM, T5, Chinchilla, Llama** — z-loss / logit z-regularization to stabilize the head and bf16 numerics.
- **Tied embeddings** — standard in Gemma, GPT-2, T5, most ≤7B models; **untied** heads common at large scale (PaLM, GPT-3 untied) where the param cost is negligible and decoupling helps.

## Common mistake

Conflating the two issues. The **softmax bottleneck** is an *expressivity* limit (rank ≤ d) independent of precision; **logit cap / z-loss** are *numerical-stability/optimization* tricks. Soft-capping does NOT fix the rank bottleneck — and capping is incompatible with vanilla FlashAttention (which never materializes the score matrix), which is why some 2024+ models drop attn-logit caps in favor of QK-norm.

## See also
- [[softmax]] — the operation whose rank and saturation behavior this brick is about
- [[z-loss-logit-stabilization]] — the logsumexp penalty alternative to tanh capping
- [[matrix-rank]] — the rank ≤ d argument is the whole bottleneck
