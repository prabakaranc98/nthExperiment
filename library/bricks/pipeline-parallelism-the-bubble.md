# Pipeline Parallelism & the Bubble

**One-liner:** Partition layers into stages across GPUs and stream micro-batches through them; the schedule (GPipe, 1F1B, interleaved, zero-bubble/DualPipe) trades activation memory against the idle "bubble" at pipeline fill/drain.

## The formula / definition

Split L layers into P stages, one stage per device. A batch is chopped into m micro-batches that flow stage→stage. Each device is idle while the pipeline fills (warmup) and drains (cooldown).

**Bubble fraction (GPipe, all forwards then all backwards):**

    bubble = (P - 1) / (m + P - 1)

So to hide the bubble you need m >> P (many micro-batches per pipeline depth). With P=8, m=8 → 47% idle; m=64 → 10%.

**Schedules:**
- **GPipe** — F-then-B. Simple, but must stash activations for *all* m micro-batches → activation memory ∝ m.
- **1F1B (PipeDream-Flush)** — interleave: once warmed up, do one forward then one backward per step. Same bubble as GPipe but activation memory ∝ P (only P in-flight), not m. Default in Megatron.
- **Interleaved 1F1B (virtual pipeline)** — give each device v non-contiguous chunks; bubble shrinks to (P-1)/(v·(m+P-1)) at the cost of v× more pipeline communication.
- **Zero Bubble (ZB-H1/H2, 2024)** — split the backward into B_input (dx) and B_weight (dW); dW has no downstream dependency, so it fills the bubble. ~0% bubble.
- **DualPipe (DeepSeek-V3, 2024)** — bidirectional schedule that overlaps forward and backward comm/compute from both pipeline ends; near-zero bubble, used to overlap all-to-all MoE comm.

## Where it appears

- **Megatron-LM** — 1F1B + interleaved is the standard PP backbone of 3D parallelism (TP×PP×DP).
- **DeepSeek-V3 / R1** — DualPipe to overlap expert-parallel all-to-all and cut bubble at 2048-GPU scale.
- **GPipe / PipeDream** — the original micro-batching and 1F1B formulations.
- **Zero Bubble PP** — adopted where activation-memory budget allows the dW split.

## Common mistake

Thinking PP splits a single forward pass across GPUs in parallel — it doesn't. Within one micro-batch the stages run *sequentially* (stage k waits for stage k-1); parallelism comes only from pipelining *different* micro-batches. Also: confusing the schedule's effect on the *bubble* (latency) with its effect on *peak activation memory* — 1F1B vs GPipe have the same bubble but very different memory.

## See also
- [[3d-nd-parallelism]] — PP is the P axis combined with TP and DP/FSDP
- [[gradient-accumulation-micro-batching]] — micro-batches are the unit that fills the pipe
- [[computation-communication-overlap]] — what zero-bubble/DualPipe exploit to hide idle time
