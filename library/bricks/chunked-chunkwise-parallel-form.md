# Chunked / Chunkwise-Parallel Form

**One-liner:** Split the sequence into chunks; compute exact intra-chunk attention quadratically and carry a summarized recurrent state across chunks linearly — blending parallel-friendly local compute with linear-time global propagation for tensor-core efficiency (FlashLinearAttention).

## The formula / definition

For a linear-attention-style recurrence S_t = S_{t-1} + k_t v_t^T, o_t = q_t S_t, split the sequence into chunks of size C. Within chunk c with local index i, decompose the output into **intra** + **inter** contributions:

- **Inter-chunk** (recurrent, carries history): O_inter = Q_c · S_{c-1}, where S_{c-1} is the running state summarizing all prior chunks. State update: S_c = S_{c-1} + K_c^T V_c (a C×d-by-d matmul, done once per chunk).
- **Intra-chunk** (parallel, exact local attention): O_intra = (tril(Q_c K_c^T)) V_c, a causal masked C×C attention block computed densely.
- O_c = O_intra + O_inter.

With decay (gated/GLA): apply per-position decay factors γ to both the carried state (S_c = diag(γ) S_{c-1} + ...) and the intra-chunk mask. Cost: O(L·C·d) for intra (quadratic only in C, not L) + O(L·d^2/C) for inter. Choosing C ~ d (e.g. 64–256) balances both and keeps matmuls on tensor cores.

## Where it appears

- **FlashLinearAttention (Yang et al., 2024)** — the canonical hardware-efficient chunkwise kernel; trades recompute vs. materializing states for memory/speed, makes linear attention actually fast on GPUs.
- **GLA / Gated Linear Attention, RetNet** — chunkwise parallel form is the *training* path (parallel over chunks), while the recurrent form is the *inference* path.
- **Mamba-2 / State Space Duality** — the SSD chunked scan is exactly this: block-diagonal quadratic intra-chunk + low-rank inter-chunk state passing.
- **DeltaNet / gated DeltaNet** — chunkwise form with a WY-representation to parallelize the delta-rule fast-weight update.

## Common mistake

Thinking it is an approximation or that it changes the result. The chunked form is *mathematically exact* — it just reorganizes the same recurrence into a parallel + recurrent split. Also: confusing it with chunked *prefill* (a serving/scheduling trick for splitting prompt tokens), which is unrelated.

## See also
- [[state-space-duality]] — Mamba-2's SSD is the same intra-quadratic / inter-recurrent decomposition
- [[gated-linear-attention-data-dependent-decay]] — the decay variant whose training path *is* this form
- [[recurrence-convolution-scan-duality]] — the broader parallel-vs-sequential equivalence this exploits
