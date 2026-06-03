# Gradient Accumulation & Micro-Batching

**One-liner:** Split a large effective batch into K micro-batches, run forward/backward on each and sum (or average) the gradients before a single optimizer step — trading wall-clock for a large batch that won't fit in memory, and deferring the all-reduce to the last micro-batch.

## The mechanism

Effective batch = micro_batch_size × K × data_parallel_world_size.

```
optimizer.zero_grad()
for i in range(K):                      # K micro-batches
    loss = forward(micro_batch[i]) / K  # scale so grads AVERAGE, not sum
    loss.backward()                     # gradients accumulate in .grad
optimizer.step()                        # one update per K micro-batches
```

Gradients land additively in `.grad` because PyTorch accumulates by default — that's why `zero_grad()` is outside the loop. Divide the loss by K (or sum-then-divide) so the accumulated gradient equals the full-batch mean gradient, matching the loss-reduction convention.

**The communication trick:** in DDP/FSDP, skip the gradient all-reduce on the first K−1 micro-batches (`model.no_sync()`); only sync on the final one. This amortizes one collective over K micro-batches instead of K collectives.

## Where it appears

- **Every large-batch pretraining run** — reaching million-token batches (GPT-3, Llama, PaLM) on memory-bound GPUs; combined with gradient checkpointing to push micro-batch size down further.
- **DeepSpeed / Megatron-LM** — `gradient_accumulation_steps` is a first-class config; pipeline parallelism *is* micro-batching (the pipeline schedule feeds micro-batches to fill the bubble).
- **GRPO / PPO RL fine-tuning** — accumulate over many rollout micro-batches to form a stable policy-gradient estimate.

## Common mistake

Forgetting BatchNorm and per-token loss normalization don't commute with accumulation. BatchNorm statistics are computed per micro-batch (K small batches ≠ one big batch) — use a norm that's batch-independent. And with token-level loss (variable-length sequences), naive `loss/K` mis-weights micro-batches with different token counts; normalize by total tokens across the accumulation window, not by K. This was the silent bug behind several "grad accum != large batch" reports in 2024.

## See also
- [[critical-batch-size-gradient-noise-scale]] — tells you when a larger effective batch stops helping
- [[gradient-checkpointing]] — the orthogonal memory lever, usually stacked with accumulation
- [[zero]] — shards optimizer state so you can also overlap the deferred all-reduce
