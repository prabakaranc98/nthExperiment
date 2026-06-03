# Computation-Communication Overlap

**One-liner:** Hide collective latency by issuing comms on separate CUDA streams that run concurrently with compute (prefetch all-gather, bucket gradient reduce-scatter, fuse TP comms into GEMMs) — the single biggest determinant of multi-GPU scaling efficiency.

## The key insight

A GPU has independent copy/comm and compute engines. If comm and compute are on different streams with no false dependency, the NCCL collective runs *while* the next GEMM executes. Effective step time is `max(T_compute, T_comm)` instead of `T_compute + T_comm`. Scaling efficiency:

    eff = T_compute / max(T_compute, T_comm)

Overlap is perfect (eff→1) only when `T_comm ≤ T_compute` AND no SM/bandwidth contention; otherwise you are *exposed* (comm-bound).

## How it's done per parallelism strategy

- **FSDP / ZeRO-3:** prefetch the next layer's all-gather (params) during current layer's forward; overlap the reduce-scatter (grads) of layer L with the backward compute of layer L-1. Bucketing groups small grads so collectives are large enough to saturate links.
- **DDP / ZeRO-2:** gradient bucketing — fire async all-reduce as each bucket fills during backprop, so reduction overlaps the rest of the backward pass.
- **Tensor parallel:** all-reduce/all-gather sits *on the critical path* of every layer. Decompose into tiles and pipeline the per-tile reduce-scatter with the next tile's matmul (e.g. Megatron's overlapped layernorm-linear, "fused" / async-TP). Hardest to hide because comm is serial with compute.
- **Pipeline parallel:** P2P send/recv of activations overlaps the next micro-batch's compute (the bubble is a separate problem).
- **Expert parallel (MoE):** overlap the all-to-all token dispatch/combine with expert GEMMs (DeepSeek-V3 DualPipe, dispatch/compute interleaving).

## Where it appears

- DeepSeek-V3 DualPipe — bidirectional pipeline schedule that fully overlaps all-to-all + PP comms with compute, near-zero exposed EP communication.
- Megatron-LM / TransformerEngine — tensor-parallel comm/GEMM overlap, sequence-parallel reduce-scatter overlap.
- PyTorch FSDP2 — explicit prefetch + reduce-scatter overlap; the default reason FSDP scales.

## Common mistake

Assuming separate streams guarantee overlap. NCCL kernels consume SMs and bandwidth, so a "concurrent" collective can *slow down* the GEMM it overlaps (contention), and any unintended dependency (a sync, a `.item()`, a profiler, or grouping params wrongly) serializes them silently. Always measure exposed comm time, not just whether streams exist.

## See also
- [[collective-communication-primitives]] — the all-gather/reduce-scatter ops being hidden
- [[fsdp]] — overlap is the core mechanism making sharded data parallel scale
- [[pipeline-parallelism-the-bubble]] — a related but distinct latency-hiding problem
