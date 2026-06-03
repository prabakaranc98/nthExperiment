# ZeRO

**One-liner:** Zero Redundancy Optimizer — shard the optimizer state (stage 1), gradients (stage 2), and parameters (stage 3) across data-parallel ranks instead of replicating them, cutting per-GPU memory ~linearly in the world size while keeping data-parallel semantics.

## The memory it eliminates

Standard data parallelism replicates *everything* on every GPU. With mixed-precision Adam, per-parameter memory is dominated by the optimizer state:

- fp16 params: 2 bytes
- fp16 grads: 2 bytes
- fp32 master params + Adam m + Adam v: 4 + 4 + 4 = 12 bytes

So ~16 bytes/param (the "16Ψ" in the paper), and the 12-byte optimizer state is the redundant bulk. For N parameters across P ranks, ZeRO partitions so each rank holds 1/P of each sharded component:

- **Stage 1 (ZeRO-1):** shard optimizer state → ~ 4Ψ + 12Ψ/P per GPU
- **Stage 2 (ZeRO-2):** + shard gradients → ~ 2Ψ + 14Ψ/P
- **Stage 3 (ZeRO-3):** + shard parameters → ~ 16Ψ/P (full linear savings)

## How comm stays cheap

Baseline DP does an all-reduce of gradients = 2Ψ traffic. ZeRO-1/2 re-express this as reduce-scatter (grads) + all-gather (updated params) = also ~2Ψ — **same comm volume, lower memory**. Stage 3 must additionally all-gather params just-in-time for each layer's forward/backward, adding ~Ψ more traffic (1.5× total), traded for the biggest memory win.

## Where it appears

- **DeepSpeed** — original implementation (Rajbhandari et al., 2020); ZeRO-Offload/Infinity push optimizer state to CPU/NVMe.
- **PyTorch FSDP** — ZeRO-3 reimagined as `FullyShardedDataParallel`; the default sharding strategy for large-model training in the PyTorch ecosystem.
- **Frontier pretraining** — combined with tensor/pipeline parallelism in 3D-parallel stacks (Megatron-DeepSpeed) to train 100B+ models.

## Common mistake

Thinking ZeRO is a *parallelism* axis like tensor or pipeline parallel — it is not. It is still pure data parallelism: every rank processes a different microbatch and computes the full model. ZeRO only changes *where the state lives*, not how the computation is split. Also: ZeRO-1/2 add almost no comm, but ZeRO-3's per-layer all-gathers can bottleneck on slow interconnects — that's the real cost of stage 3.

## See also
- [[fsdp]] — PyTorch's native ZeRO-3; same idea, different name
- [[distributed-training]] — ZeRO is the memory-sharding leg of the parallelism taxonomy
- [[tensor-parallel]] — orthogonal axis combined with ZeRO in 3D parallelism
