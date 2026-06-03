# Sequence Packing & Document Attention Masking

**One-liner:** Concatenate variable-length documents into fixed-length training sequences to eliminate padding waste, with a block-diagonal attention mask (and per-document position reset) so tokens never attend across document boundaries.

## The definition

Pack documents d₁, d₂, ... greedily (or via bin-packing) into a sequence of length L. Build segment IDs s ∈ {0,1,...} marking which document each token belongs to. The attention mask combines causality with same-document membership:

    M[i,j] = 1  iff  j ≤ i  AND  s[i] == s[j]

so the logits become block-diagonal-lower-triangular: token i attends only to earlier tokens *in its own document*. RoPE/absolute positions are also reset per document (positions restart at 0 each boundary), and the loss is masked so a document's first token has no cross-document prediction target.

## Where it appears

- **All modern pretraining stacks (Llama 3, GPT-NeoX, Megatron, T5)** — packing is the default; "naive packing" (concat without masking) was long tolerated but shown harmful for long context.
- **FlashAttention `varlen` / `cu_seqlens` API** — passes cumulative sequence lengths so the fused kernel computes block-diagonal attention with zero padding compute. This is how packing is made cheap in practice.
- **xformers `BlockDiagonalMask`, JAX `make_attention_mask` + segment_ids** — framework primitives for the same construct.
- **Best-fit-decreasing bin packing (e.g. "Fewer Truncations Improve LM", 2024)** — smarter packing reduces document *truncation* across sequence boundaries, improving downstream quality.

## Common mistake

Concatenating documents but forgetting the intra-document mask and position reset ("contamination"). Tokens then attend backward into unrelated documents, and positions drift across boundaries — corrupting long-context behavior and inflating effective context. With FlashAttention you must pass `cu_seqlens` (varlen), not just a packed tensor, or the kernel treats the whole sequence as one document.

## See also
- [[flash-attention]] — its varlen/`cu_seqlens` interface is the efficient implementation path
- [[rope]] — positional encoding that must be reset per document when packing
- [[grouped-document-aware-loss-masking-token]] — the complementary loss-side masking at document boundaries
