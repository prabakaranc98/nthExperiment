# NoPE / Positional-Scheme Length Generalization

**One-liner:** Decoder-only transformers can encode and extrapolate token order with NO explicit positional encoding — the causal mask alone breaks permutation symmetry — reframing the choice of positional scheme (RoPE vs ALiBi vs NoPE) as a length-generalization knob, not a correctness requirement.

## The key insight

A bidirectional transformer is permutation-equivariant, so it needs positional info to know order. But a **causal** decoder is not: each position i attends only to {0..i}, so position 0 sees 1 token, position i sees i+1 tokens. Attention can read this asymmetry — counting how many tokens are visible recovers absolute position implicitly. Formally (Kazemnejad et al., NeurIPS 2023), a NoPE layer can express a position-counting function and thereby simulate both absolute and relative encodings; positional structure emerges in the learned attention/values rather than being injected.

Causal mask M (lower-triangular): softmax(QKᵀ/√d + M)V, M_{ij} = 0 if j ≤ i else −∞. No position term added anywhere.

## Where it appears

- **Kazemnejad et al. 2023 ("The Impact of Positional Encoding on Length Generalization")** — NoPE beats RoPE/ALiBi/T5-bias on length-generalization for small decoder-only reasoning tasks; the headline result.
- **Frontier long-context models** — production systems do NOT use pure NoPE; the lesson is absorbed via RoPE base/frequency scaling (YaRN, NTK-aware) since NoPE alone underperforms at scale on natural language.
- **Hybrid attention–SSM stacks** — SSM/Mamba layers carry order recurrently, so attention layers can run NoPE-style; some hybrids drop RoPE in interleaved attention blocks.
- **Length-generalization analyses** — used as the baseline showing extrapolation failure is a property of the positional scheme, not the transformer.

## Common mistake

Concluding "positional encodings are unnecessary, just use NoPE." NoPE only works in the **causal** setting (encoders still need positions), it learns position implicitly so it is sensitive to train-length distribution, and at real LLM scale carefully-scaled RoPE generally extrapolates better than pure NoPE. NoPE is a clarifying lens, not a free lunch.

## See also
- [[rope]] — the dominant explicit scheme NoPE is contrasted against
- [[rope-context-extension]] — how production models actually get length generalization (base scaling/YaRN) instead of NoPE
- [[hybrid-attention-ssm-architectures]] — recurrent layers supply order, freeing attention from positional encoding
