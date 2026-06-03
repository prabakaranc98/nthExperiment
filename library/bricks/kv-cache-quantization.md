# KV-Cache Quantization

**One-liner:** Store cached K/V in low precision (FP8/INT8/INT4) with the right granularity — per-channel for keys, per-token for values — to halve or quarter KV memory and bandwidth, extending context and batch size at near-lossless quality.

## The key insight

Decode is bottlenecked by *reading* the KV cache (bandwidth-bound), and the cache dwarfs the weights at long context. Quantizing it cuts both footprint and bytes-moved per step. The catch is that K and V have different outlier structure, so they need different scaling axes:

- **Keys**: outliers are concentrated in fixed *channels* (feature dims) → quantize **per-channel** (one scale per dim, shared across tokens). Per-token scaling on keys is what kills accuracy.
- **Values**: no channel structure → quantize **per-token** (one scale per token, computed at append time, streaming-friendly).

INT4 group-quant of values, scale s per group g of size G:
  s = (max|v| in g) / (2^{b−1} − 1),  q = clamp(round(v / s), −2^{b−1}, 2^{b−1}−1),  v̂ = s·q

Per-channel key quant means the running max along a channel must be tracked as tokens stream in (or a residual of recent unquantized tokens is kept in fp16). Total cache bytes scale by (b/16) vs fp16; INT4 → 4×, FP8/INT8 → 2×.

## Where it appears

- **KIVI** — asymmetric 2-bit: per-channel keys + per-token values, keeping a small fp16 residual window of recent tokens; ~2.6× less KV memory, larger batch.
- **KVQuant** — per-channel keys *before* RoPE, non-uniform (NUQ) levels, dense-and-sparse outlier split; pushes to ~3-bit with tiny perplexity loss.
- **vLLM / TensorRT-LLM** — FP8 (E4M3) and INT8 KV-cache kernels as a serving flag; doubles effective KV capacity on Hopper/Blackwell.
- **Hadamard/QuaRot-style rotation** — rotate K/V to spread outliers across channels so flat per-token INT4 becomes viable.

## Common mistake

Applying per-tensor or per-token scaling uniformly to *both* K and V. Keys have per-channel outliers (often amplified by RoPE), so a per-token key scale lets one bad channel crush the rest — accuracy collapses. The asymmetry (per-channel keys, per-token values) is the whole trick, not an optimization.

## See also
- [[kv-cache]] — the bandwidth/memory bottleneck this directly attacks
- [[activation-outliers-smoothquant]] — same outlier problem; motivates per-channel scaling
- [[hadamard-rotation-based-quantization]] — rotate to flatten outliers before low-bit KV quant
