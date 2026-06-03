# Sequence / Context Parallelism

**One-liner:** Shard the sequence (token) dimension across GPUs so each device holds only part of the context; Ring Attention streams K/V around the ring and Ulysses does an all-to-all over heads to compute *exact* attention over million-token sequences that no single GPU could fit.

## The key insight

Activations scale with sequence length S, not just model size, so long context is memory-bound on S. Split the S axis across P devices (each gets S/P tokens). The MLP and norm layers are embarrassingly parallel per token. Attention is the only coupled op — every query must attend to *all* keys — so the two schemes differ in how they move K/V:

**Ring Attention (Liu et al., 2023):** each device keeps its local Q and a block of K/V, computes a partial attention, then passes its K/V block to the next device in a ring while receiving the previous one. After P steps every Q has seen every K/V. Uses the **online-softmax** recurrence (running max m, normalizer ℓ) to merge partial outputs exactly — no approximation. Communication overlaps with compute; memory per device is O(S/P), enabling near-arbitrary context with enough devices.

**Ulysses (DeepSpeed, Jacobs et al., 2023):** before attention, an **all-to-all** reshards from sequence-parallel to head-parallel layout — each device gets all tokens for a subset of heads, runs standard (Flash) attention locally on full-length sequences, then a second all-to-all reshards back to sequence-parallel for the MLP. Comm volume per step is independent of P (scales with hidden size), but parallelism is capped by the number of heads.

## Where it appears

- **Ring Attention / Blockwise** — Gemini-class and Llama-class long-context (≥1M tokens); pairs naturally with FlashAttention as the local kernel.
- **DeepSpeed-Ulysses / Megatron context parallelism** — training stacks; Megatron-CP combines ring CP with tensor parallel inside a node.
- **Llama 3 405B** — used CP for 128K-token continued pretraining; **4D/5D parallelism** lists CP as an axis alongside DP/TP/PP/EP.

## Common mistake

Confusing it with tensor parallelism. TP shards the *hidden/head* dimension of weights; SP/CP shards the *token* dimension of activations and inputs. They are orthogonal and routinely composed. Also: Ring Attention is exact (online softmax), not a sparse/approximate attention — the only catch is load imbalance under causal masking, which needs zig-zag/striped token assignment to fix.

## See also
- [[ring-attention-blockwise-distributed-attention]] — the canonical ring-streaming K/V scheme
- [[online-softmax-safe-softmax-recurrence]] — the exact merge of partial attention blocks
- [[3d-nd-parallelism]] — CP/SP as one orthogonal axis composed with DP/TP/PP/EP
