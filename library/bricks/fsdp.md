# FSDP — Fully Sharded Data Parallel

**One-liner:** PyTorch's native ZeRO-3 — shards params, grads, and optimizer state across N ranks, all-gathers each layer's full weights just-in-time for compute, then reshards immediately, trading communication for a ~N× memory cut.

## The mechanics

Wrap the model in nested *FSDP units* (typically one per transformer block). Each rank stores only 1/N of every unit's parameters as a flat shard. Per step:

- **Forward:** before unit `i` runs, `all_gather` its full params from all ranks → compute → free the gathered params (reshard).
- **Backward:** `all_gather` params again → compute grads → `reduce_scatter` grads so each rank ends up with its 1/N grad shard → free params.
- **Optimizer:** each rank updates only its 1/N of params/optimizer state locally; no broadcast needed.

Per-rank memory ≈ (P + 2P_optim)/N + peak activations + (size of the single largest gathered unit). Communication volume ≈ 1.5× pure DDP (gather fwd + gather bwd + reduce-scatter) vs DDP's single all-reduce.

## FSDP2 (2024+)

The rewrite (`fully_shard`) replaces `FlatParameter` with per-parameter `DTensor` sharding: cleaner state_dict, composes with `torch.compile`, tensor/pipeline parallel, and FP8. `HYBRID_SHARD` shards within a node and replicates across nodes (DDP between nodes) to cut inter-node traffic. Overlap is achieved by *prefetching* unit `i+1`'s all-gather during unit `i`'s compute.

## Where it appears

- **Llama / OLMo / open LLM pretraining** — FSDP (often HYBRID_SHARD) is the default sharding strategy in `torchtitan` and most HF/Accelerate training stacks.
- **DeepSpeed ZeRO-3** — the same algorithm in a different framework; FSDP is PyTorch's first-party answer to it.
- **QLoRA + FSDP** — shard a quantized base across GPUs for full-context fine-tuning of 70B+.

## Common mistake

Thinking FSDP shards activations — it does not. It shards *parameters, gradients, optimizer state*. Activation memory still scales with batch×seq and is FSDP's real OOM source at long context; you combine it with [[gradient-checkpointing]]. Also: wrapping granularity matters — wrapping the whole model as one unit gathers all params at once and defeats the memory savings.

## See also
- [[zero]] — FSDP is ZeRO-3; the stage taxonomy (1/2/3) maps to what FSDP shards
- [[tensor-parallel]] — orthogonal axis; combined with FSDP for 2D/3D parallelism on large models
- [[gradient-checkpointing]] — the complementary trick for the activation memory FSDP leaves untouched
