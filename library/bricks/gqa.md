# Grouped-Query Attention (GQA / MQA)

**One-liner:** Share K/V projections across groups of query heads — MQA uses one shared K/V head, GQA uses G groups — shrinking the KV cache by H/G× at modest quality cost; GQA is the modern default.

## The definition

Standard multi-head attention (MHA) with H heads has H separate Q, K, V projections. GQA partitions the H query heads into G groups; each group shares one K head and one V head:

- **MHA:** G = H (every query head has its own K/V) — full quality, full cache
- **MQA:** G = 1 (all query heads share a single K/V) — smallest cache, biggest quality hit
- **GQA:** 1 < G < H (interpolates) — typically G = 8

KV cache size ∝ G·d_head per layer per token (was H·d_head). With H=64, G=8 that is an **8× cache reduction**. The shared K/V are broadcast (repeated H/G times) before the QKᵀ score computation, so attention math is otherwise unchanged.

## Why it matters

Autoregressive decoding is **memory-bandwidth-bound**: each token must read the entire KV cache from HBM. Cache size scales with sequence length, batch, and layers, so it dominates memory at long context and caps batch size. Shrinking K/V heads shrinks the bytes moved per step → higher throughput and longer feasible context, with FLOPs essentially flat (Q heads unchanged).

## Where it appears

- **GQA paper (Ainslie et al., 2023)** — introduced GQA; showed MHA→GQA can be *uptrained* from an existing checkpoint with ~5% of pretraining compute by mean-pooling K/V heads into groups
- **Llama 2 70B / Llama 3, Mistral 7B, Gemma, Qwen2** — all ship GQA (commonly G=8) as the production default
- **PaLM, Falcon, original Whisper decoder** — used MQA (G=1) for decode speed
- **MLA (DeepSeek-V2/V3)** — the successor idea: compress K/V into a low-rank latent instead of dropping heads, cutting cache further while recovering MHA-like quality

## Common mistake

Thinking GQA reduces compute or parameters meaningfully. The win is **KV-cache memory and decode bandwidth**, not FLOPs — query heads (and thus the QKᵀ·V work) are untouched. It is an inference-memory optimization, and at small batch / short context the speedup can be negligible.

## See also
- [[kv-cache]] — the exact memory bottleneck GQA shrinks
- [[flash-attention]] — orthogonal: GQA cuts cache bytes, FlashAttention cuts attention HBM traffic; stacked in every modern stack
- [[inference-and-serving]] — GQA is a core lever for throughput and max batch size in serving
