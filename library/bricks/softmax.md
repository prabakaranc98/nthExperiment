# Softmax

**One-liner:** Converts a vector of real-valued logits to a probability distribution; temperature τ controls the sharpness.

## The formula

softmax(zᵢ) = exp(zᵢ/τ) / Σⱼ exp(zⱼ/τ)

- τ = 1: standard
- τ → 0: approaches argmax (deterministic, peaked)
- τ → ∞: approaches uniform distribution (flat, maximum entropy)

## Where it appears

- **Attention** — softmax(QKᵀ/√dₖ) · V; √dₖ is the temperature
- **Next-token prediction** — final layer logits → probabilities over vocabulary
- **Contrastive learning** — InfoNCE loss uses softmax over negatives; temperature is a hyperparameter
- **GRPO / PPO** — policy outputs probabilities via softmax

## Common mistake

Forgetting the numerical stability trick: subtract max(z) before exponentiating. Without it, exp() overflows for large logits. All production implementations do `softmax(z - max(z))`.

## See also
- [[cross-entropy]] — what you minimize using softmax outputs
- [[gqa]] — softmax in grouped query attention
- [[flash-attention]] — online softmax is the key algorithmic trick
