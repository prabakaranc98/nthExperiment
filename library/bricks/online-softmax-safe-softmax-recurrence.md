# Online Softmax / Safe-Softmax Recurrence

**One-liner:** A single-pass, numerically-stable recurrence that maintains a running max and running denominator so softmax (and its weighted output) can be computed over a streamed sequence of scores without ever materializing the full row — the primitive that makes FlashAttention exact and overflow-free.

## The recurrence

Naive softmax needs the full row to subtract the max for stability. Online softmax streams it. Process scores in blocks; maintain running max m and running sum (normalizer) ℓ. For a new block with local max mₖ:

    m_new = max(m_old, mₖ)
    ℓ_new = ℓ_old · exp(m_old − m_new)  +  Σ_{i∈block} exp(xᵢ − m_new)

The `exp(m_old − m_new)` factor **rescales the old partial sum** to the new max — both ≤ 0 in the exponent, so no overflow. In attention, the output accumulator O is rescaled by the same factor:

    O_new = O_old · exp(m_old − m_new)  +  Σ_i exp(xᵢ − m_new) · vᵢ

Final divide by ℓ once at the end. One pass, O(1) extra state per row, mathematically identical to two-pass safe softmax.

## Where it appears

- **FlashAttention (1/2/3)** — the entire algorithm is this recurrence over K/V tiles in SRAM; lets attention avoid writing the N×N score matrix to HBM
- **Ring / blockwise / sequence-parallel attention** — each device streams a shard of K/V and combines partial (m, ℓ, O) statistics across ranks via the same merge rule
- **Streaming / chunked-prefill decode** — extend the running max/sum as new KV blocks arrive without recomputing earlier softmax
- **Original "Online normalizer calculation for softmax"** (Milakov & Gimelshein, 2018) — the standalone formulation predating its use in attention

## Common mistake

Forgetting to rescale the **output accumulator** (and any other running statistic) by the same `exp(m_old − m_new)` factor when the max updates. Rescaling only ℓ but not O gives silently wrong attention outputs — a bug that passes shape checks but fails numerics.

## See also
- [[flash-attention]] — built directly on this recurrence to make attention IO-aware
- [[softmax]] — the function being computed; this is its stable streaming form
- [[ring-attention-blockwise-distributed-attention]] — merges per-shard (m, ℓ, O) using the same rule across devices
