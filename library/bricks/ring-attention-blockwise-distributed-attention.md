# Ring Attention / Blockwise Distributed Attention

**One-liner:** Shard the sequence across devices and pass KV blocks hop-by-hop around a ring while each device computes its local attention block — exact attention with per-device memory independent of total length, enabling million-token context (the backbone of context/sequence parallelism).

## The key insight

Blockwise attention (Rabbani & Hoffmann; flash-style online softmax) lets you compute exact attention one KV block at a time, keeping running max `m` and normalizer `l` per query row. Ring Attention (Liu, Zaharia & Abbeel, 2023) distributes this across `P` devices:

- Device `i` holds query block `Q_i` (and its `K_i, V_i`) — sequence sharded along length.
- Loop `P` steps. At each step, every device computes the partial attention of its local `Q_i` against the KV block it currently holds, updating `(O_i, m_i, l_i)` via online softmax.
- Meanwhile it sends its current KV block to neighbor `i+1` and receives the next from `i-1` (a ring `all-gather` of KV, one hop per step).

Because the KV send/recv is overlapped with the local block compute, communication is fully hidden when the per-block FLOPs exceed the transfer time. Per-device activation memory is O(N/P) for Q,O and O(N/P) for the resident KV block — so total supported `N` scales linearly with device count, in principle "near-infinite."

```
O_i, m_i, l_i = 0, -inf, 0
KV = (K_i, V_i)
for step in range(P):
    send(KV -> i+1) || recv(next_KV <- i-1)        # overlapped comm
    O_i, m_i, l_i = online_softmax_update(Q_i, KV, O_i, m_i, l_i)  # local compute
    KV = next_KV
O_i /= l_i
```

Causal masking creates load imbalance (early ranks do less work); **Striped/Zigzag Attention** permutes token-to-device assignment so each device gets a balanced mix of causal positions.

## Where it appears

- **Gemini 1.5 / long-context frontier LLMs** — context parallelism over 1M+ tokens during training and prefill builds directly on ring/blockwise attention.
- **Megatron-LM & DeepSpeed-Ulysses** — context-parallel implementations; Ulysses uses all-to-all on heads as an alternative to the ring, often combined with ring for hybrid CP.
- **Blockwise Parallel Transformer (BPT)** — fuses ring attention with blockwise feedforward to also shard MLP memory.
- **TransformerEngine / FlashAttention CP** — ship ring + zigzag kernels for production context parallelism.

## Common mistake

Thinking it changes the math or approximates attention — it computes the *exact* same output as full attention; only the memory layout and communication pattern change (like FlashAttention, but across devices instead of across SRAM tiles). The second trap: ignoring causal load imbalance and getting poor utilization because you used naive contiguous sharding instead of striped/zigzag assignment.

## See also
- [[sequence-context-parallelism]] — the parallelism paradigm ring attention implements
- [[online-softmax-safe-softmax-recurrence]] — the running-max/normalizer recurrence that makes blockwise exactness possible
- [[flash-attention]] — the single-device tiling analogue ring attention generalizes across the network
