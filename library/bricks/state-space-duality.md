# State-Space Duality (SSD / Mamba-2)

**One-liner:** Framework proving selective SSMs and a causal-masked form of linear attention are two views of the same structured (semiseparable) matrix transform, so Mamba-2's scan can be reformulated as block matmuls that saturate tensor cores.

## The key insight

A scalar-decay selective SSM (state $h_t = A_t h_{t-1} + B_t x_t$, $y_t = C_t^\top h_t$, with $A_t = a_t I$) computes a sequence transform $Y = M X$ whose matrix is *1-semiseparable* and lower-triangular:

$$M_{ij} = C_i^\top \Big(\prod_{k=j+1}^{i} a_k\Big) B_j \quad (i \ge j),\ 0\ \text{otherwise}$$

That is exactly **causal linear attention with a data-dependent decay mask**: $M = L \odot (C B^\top)$, where $L_{ij}=\prod_{k=j+1}^i a_k$ is the cumulative-decay lower-triangular mask. Two algorithms for one matrix:
- **Linear / recurrent form:** materialize the state, $O(N)$ time, $O(1)$-per-step memory (decode).
- **Quadratic / attention form:** materialize $M$, $O(N^2)$, but pure matmul (short seqs).

SSD uses the **chunked (block-decomposition) algorithm**: split into chunks, compute intra-chunk via the quadratic form (tensor cores) and inter-chunk via the recurrence on chunk-summary states — $O(N L)$ for chunk length $L$, mostly bf16 matmuls. Mamba-2 also relaxes Mamba-1's diagonal $A$ to scalar-times-identity $a_t I$, enabling this and ~2-8x faster training.

## Where it appears

- **Mamba-2 (Dao & Gu, 2024)** — the SSD layer; bigger state dim (N=64-256), multi-head SSD analogous to MHA, parallelizable like attention.
- **Hybrid attention-SSM stacks (Jamba, Zamba, NVIDIA Nemotron-H, Falcon-Mamba)** — SSD blocks interleaved with a few softmax-attention layers.
- **Linear-attention lineage (GLA, mLSTM/xLSTM, RetNet, DeltaNet)** — all instances of "structured-matrix sequence mixer"; SSD unifies their chunkwise-parallel kernels.

## Common mistake

Thinking SSD makes Mamba *equal* to softmax attention. The duality is with **linear** attention (no softmax) under a specific decay structure — the mask $L$ is a cumulative product of scalars, not arbitrary. You trade softmax's content-based selectivity for a fixed-size compressible state; expressivity (e.g. exact long-range associative recall) genuinely differs.

## See also
- [[selective-state-space-models-mamba]] — Mamba-1, the selective SSM that SSD reformulates
- [[linear-attention]] — the other half of the duality; causal masked form
- [[chunked-chunkwise-parallel-form]] — the algorithm that realizes SSD on tensor cores
