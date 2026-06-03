# Data Parallelism (DDP)

**One-liner:** Replicate the full model on every one of N GPUs, split each global batch into N per-GPU shards, and `all_reduce` the gradients every step so all replicas stay bit-identical — the baseline every training stack starts from.

## The mechanics

Every rank holds an identical copy of params, grads, and optimizer state. Per step:

- Each rank `r` runs forward + backward on its own shard `B/N` of the batch → local grad `g_r = (1/|B_r|) Σ ∇ℓ`.
- `all_reduce(g_r, op=SUM)` then divide by N (or average directly) → every rank now holds the same global mean gradient `ḡ = (1/N) Σ_r g_r`.
- Each rank applies the *same* optimizer step locally → replicas stay identical, no broadcast needed.

Mathematically the all-reduced gradient equals the gradient of the full batch B (gradient is linear in the sum over examples), so DDP with global batch B is exactly equivalent to single-GPU training on B — same loss curve, modulo float non-associativity.

## The overlap trick

Naïve DDP all-reduces once after the whole backward finishes — comm and compute serialize. Real DDP (`torch.nn.parallel.DistributedDataParallel`) registers autograd hooks, **buckets** params (default ~25 MB), and fires each bucket's `all_reduce` as soon as its grads are ready, overlapping comm with the rest of the backward pass. Ring all-reduce moves ~`2P` bytes/rank regardless of N (bandwidth-optimal); cost is roughly constant in world size on a good interconnect.

## Where it appears

- **Every training stack's default** — PyTorch DDP / `torchrun`, JAX `pmap`, Horovod; the leg you add first before any model sharding.
- **ZeRO / FSDP** — both are *data parallelism* that shard the replicated state instead of duplicating it; DDP is the un-sharded special case.
- **3D / nD parallelism** — DDP is the outer axis composed with tensor + pipeline parallel; `HYBRID_SHARD` does FSDP within a node and DDP across nodes.

## Common mistake

Assuming DDP scales batch for free. It cuts step *time*, not memory: every GPU still stores the full model + Adam state, so DDP alone never fits a model that doesn't fit on one GPU — that's what ZeRO/FSDP/tensor-parallel are for. Also: keep the loss a per-example mean (DDP averages grads across ranks), and remember the effective global batch is `B×N`, so the LR/warmup schedule must be re-tuned, not copied from the single-GPU run.

## See also
- [[zero]] — DDP that shards optimizer/grad/param state to kill the per-GPU memory it replicates
- [[fsdp]] — PyTorch's ZeRO-3; the sharded successor to plain DDP
- [[collective-communication-primitives]] — `all_reduce`/ring-reduce is the operation DDP is built on
