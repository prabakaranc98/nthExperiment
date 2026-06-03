# FlashAttention

**One-liner:** IO-aware exact attention via tiling — computes the same result as standard attention but reads/writes HBM O(N) times instead of O(N²), giving 2-4× wall-clock speedup.

## The problem it solves

Standard attention materializes the full N×N attention matrix in HBM (GPU RAM):
1. Write S = QKᵀ/√d to HBM: O(N²) writes
2. Compute P = softmax(S), write to HBM: O(N²) writes
3. Read P, compute O = PV: O(N²) reads

For N=128K, this is ~128GB of unnecessary HBM traffic. The computation is memory-bound, not compute-bound.

## The key insight

Tile the computation: process blocks of Q, K, V that fit in on-chip SRAM (shared memory). Use the **online softmax** trick (Milakov & Gimelshein) to compute the running softmax normalization without seeing the full row.

Online softmax: maintain running max m and normalizer ℓ per row. As new blocks arrive, update: m_new = max(m, max(block)), ℓ_new = ℓ·exp(m-m_new) + exp(block-m_new). Rescale the output accumulator accordingly.

Result: attention computed block-by-block entirely in SRAM; HBM reads/writes reduced to O(N).

## The versions

- **FlashAttention-1** ([arXiv 2205.14135](https://arxiv.org/abs/2205.14135)) — the original; ~2× speedup
- **FlashAttention-2** ([arXiv 2307.08691](https://arxiv.org/abs/2307.08691)) — better parallelism; 2-4× over standard
- **FlashAttention-3** (2024) — FP8, async overlap with Tensor Cores; H100-optimized

## Where it appears

Every frontier LLM training and inference stack. Required reading for any systems work. The technique that made long-context transformers economically feasible.

## Common mistake

Thinking FlashAttention changes the output. It does not — it computes the *exact* same attention output, just with a different memory access pattern. It is not an approximation.

## See also
- [[roofline]] — FlashAttention moves attention from memory-bound to compute-bound
- [[kv-cache]] — FlashAttention reduces memory during training; KV cache is the inference bottleneck
