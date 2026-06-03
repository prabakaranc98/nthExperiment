# Hardware-Aware Parallel Scan

**One-liner:** A work-efficient associative (Blelloch) prefix scan fused into a single CUDA kernel that keeps SSM recurrent states in SRAM/registers — turning Mamba's selective, time-varying recurrence (which can't be a convolution) into a memory-bound op that runs in parallel across the sequence.

## The key insight

A linear recurrence h_t = A_t h_{t-1} + B_t x_t is associative, so it's a *scan* over the binary operator that composes affine maps:

    (A1, b1) ⊕ (A2, b2) = (A2·A1,  A2·b1 + b2)

Associativity means a sequential O(L) recurrence can be evaluated as a parallel prefix scan in O(log L) depth (Blelloch: up-sweep reduce + down-sweep). Selective SSMs (Mamba) make A_t, B_t, C_t *input-dependent*, which kills the global-convolution / FFT trick that LTI SSMs (S4) used — so you're forced back to a scan. The win comes from being hardware-aware:

- Load A_t, B_t, C_t into SRAM, do the discretization + scan there, never materializing the full (B, L, D, N) state tensor in HBM (it's O(N) bigger than the input).
- Fuse discretization, scan, and the C_t·h_t output projection into one kernel; recompute intermediate states in the backward pass (gradient checkpointing) instead of storing them.
- The op is memory-bound, so kernel fusion + SRAM residency is most of the speedup, not raw FLOP reduction.

## Where it appears

- **Mamba / Mamba-2** — the original "selective scan" CUDA kernel; Mamba-2 reframes it as state-space duality so the scan becomes a chunked matmul that hits tensor cores.
- **Chunked / chunkwise-parallel forms** — split sequence into chunks: scan *within* chunk, recurrence *across* chunk boundaries; standard in GLA, RWKV-6/7, DeltaNet.
- **Associative scan primitives** — `torch.associative_scan`, JAX `lax.associative_scan`, and Triton implementations now ship the algorithm without a hand-written CUDA kernel.

## Common mistake

Thinking the parallel scan reduces FLOPs vs. the sequential recurrence — it doesn't (it's *more* total work, ~2x). It wins by exposing parallelism across the sequence and avoiding HBM round-trips, just like FlashAttention. The other error: assuming you still need it for LTI SSMs — those collapse to a global convolution (FFT), and the scan only becomes mandatory once the dynamics are made *selective* (input-dependent).

## See also
- [[selective-state-space-models-mamba]] — the architecture this kernel was built for
- [[recurrence-convolution-scan-duality]] — why selectivity forces scan over convolution
- [[kernel-fusion]] — the fuse-into-one-kernel, keep-state-in-SRAM principle
