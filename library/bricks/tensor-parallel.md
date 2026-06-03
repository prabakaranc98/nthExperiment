# Tensor Parallelism

**One-liner:** Intra-layer model parallelism (Megatron-LM): shard individual weight matrices across GPUs so each device computes a slice of every layer, fused with collective communication to keep the math exact.

## The key insight

Partition matmuls so the collective is cheap. For an MLP `Y = GeLU(XA)·B`:
- **Column-parallel A**: split `A = [A₁ | A₂]` by columns → each GPU computes `GeLU(X·Aᵢ)` independently (GeLU is elementwise, no comm).
- **Row-parallel B**: split `B = [B₁; B₂]` by rows → each GPU computes a partial `Yᵢ`, then **all-reduce** to sum: `Y = ΣᵢYᵢ`.

This gives one all-reduce per MLP in the forward (and one in the backward). Attention is parallelized over heads: each GPU owns a subset of heads (column-split QKV projection), then row-parallel output projection + all-reduce. So **2 all-reduces per transformer layer per direction** (f/g conjugate operators: f = identity fwd / all-reduce bwd, g = all-reduce fwd / identity bwd).

## Where it appears

- **Megatron-LM (Shoeybi 2019)** — the canonical formulation; column-then-row sharding to minimize collectives.
- **Sequence parallelism (Korthikanti 2022)** — extends TP to shard LayerNorm/dropout activations along the sequence dim; all-reduce becomes reduce-scatter + all-gather, cutting activation memory.
- **3D parallelism (Megatron-Turing, GPT-4-scale)** — TP *within* a node (NVLink), pipeline parallel *across* nodes, data parallel on top. TP degree usually ≤ 8 (one node).
- **vLLM / TensorRT-LLM inference** — TP shards weights across GPUs to fit large models and split per-token compute.

## Common mistake

Scaling TP across node boundaries. TP issues an all-reduce on the critical path *every layer*, so it is bandwidth-bound — viable only over fast intra-node links (NVLink/NVSwitch). Past TP≈8 the inter-node all-reduce latency dominates; use pipeline or data parallelism for the rest. Also: TP is *not* a memory-only trick like ZeRO — it shards compute too, and the collectives are exact (no approximation).

## See also
- [[fsdp]] — shards params/grads/optimizer state for memory; composes with TP in 3D parallelism
- [[zero]] — data-parallel memory sharding, the orthogonal axis to TP
- [[roofline]] — TP's all-reduces make layers communication/bandwidth-bound
