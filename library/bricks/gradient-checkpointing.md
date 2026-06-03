# Gradient Checkpointing

**One-liner:** Recompute activations during the backward pass instead of storing them — trade ~1 extra forward pass for an O(√L) (or better) reduction in activation memory, the dominant memory cost of training long sequences.

## The key insight

Backprop needs each layer's input activations to compute its gradient, so naive training stores activations for all L layers: memory O(L). Gradient checkpointing saves activations only at a subset of "checkpoint" boundaries and discards the rest. During backward, it re-runs the forward pass *segment-by-segment* from the nearest saved checkpoint to regenerate the dropped activations on demand.

Classic result (Chen et al., 2016, "Training Deep Nets with Sublinear Memory Cost"): with √L evenly spaced checkpoints over an L-layer chain, activation memory drops from O(L) to **O(√L)** at the cost of **one extra forward pass** (~33% more compute, since backward ≈ 2× forward).

```
Forward:   store activation only at checkpoints; drop intermediates
Backward (per segment, from last checkpoint forward):
    recompute intermediate activations
    backprop through the segment using them
    free them again
```

## Where it appears

- **PyTorch** — `torch.utils.checkpoint.checkpoint` / `checkpoint_sequential`; `use_reentrant=False` is the modern non-reentrant variant (composes with autograd, AMP, RNG handling). HF Transformers exposes `gradient_checkpointing_enable()`.
- **Long-context / large-model training** — standard for fitting bigger batches or longer sequences; routinely stacked with FSDP/ZeRO and mixed precision.
- **Selective activation checkpointing** (Megatron-LM, Korthikanti et al. 2022) — checkpoint only the memory-heavy ops (e.g. attention) rather than whole layers, recovering most memory savings with far less recompute.
- **FlashAttention** — its backward recomputes the attention matrix from saved softmax stats; a hand-tuned, fused instance of the same recompute-don't-store principle.

## Common mistake

Thinking it saves *parameter* or *optimizer-state* memory. It does not — it only cuts **activation** memory. If your OOM is from optimizer states (Adam's 2× moments) or model weights, use ZeRO/FSDP, not checkpointing. Also: combine carefully with dropout/other RNG ops — the recomputed forward must reuse the *same* random seed (non-reentrant `checkpoint` handles this; naive reimplementations silently corrupt gradients).

## See also
- [[backpropagation]] — checkpointing is a memory/compute schedule for the backward pass
- [[mixed-precision]] — orthogonal activation-memory saver, almost always used together
- [[fsdp]] — shards weights/optimizer state; complementary axis of memory reduction
