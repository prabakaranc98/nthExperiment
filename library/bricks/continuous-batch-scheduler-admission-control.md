# Continuous-Batch Scheduler & Admission Control

**One-liner:** The per-iteration serving control loop that picks which running/waiting requests to prefill vs decode, how to chunk and pack them into a token budget, when to admit, preempt, or swap KV — subject to a KV-memory constraint and per-request SLOs (TTFT/TPOT) — the layer where chunked prefill and prefill/decode disaggregation are actually realized.

## The control loop (per step)

```
loop every iteration:
  free = total_kv_blocks − used_kv_blocks
  # 1. ADMIT from waiting queue while KV fits + token budget left
  while waiting and can_allocate(req.kv_need) and budget_left > 0:
      admit(req); reserve_kv(req)
  # 2. SCHEDULE: fill token budget B (e.g. 2048) per step
  batch = pick(running, prefill_chunks, waiting,
               by={prio, SLO_slack, FCFS, prefix-hit})
  # 3. PREEMPT under KV pressure (decode grows cache unboundedly)
  while used_kv > limit:
      victim = lowest_priority_running()   # often newest / LIFO
      recompute or SWAP victim.kv -> CPU/host  # restore later
  step(batch)  # one fused forward over mixed prefill+decode tokens
```

Key constraint: decode KV grows every step, so admission is a bin-packing problem under a moving memory budget, not a one-time decision.

## Chunked prefill & disaggregation

- **Chunked prefill (Sarathi-Serve, OSDI'24):** split a long prompt into chunks and co-schedule each chunk with ongoing decodes in one batch, hitting the token budget B. Stops a long prefill from head-of-line-blocking decodes; trades a little TTFT for stable TPOT (no decode stalls).
- **Disaggregation (DistServe, Mooncake 2024):** run prefill and decode on *separate* GPU pools so the scheduler tunes batch size for each phase's bottleneck independently; the loop above splits into a prefill scheduler + a decode scheduler with a KV transfer between them.

## Where it appears

- **vLLM** — `Scheduler` with running/waiting/swapped queues, token budget, chunked prefill on by default (2024+), recompute-or-swap preemption.
- **SGLang** — RadixAttention-aware admission: prioritizes requests with cached prefix hits to skip prefill.
- **Sarathi-Serve / TensorRT-LLM in-flight batching** — stall-free batching via prefill chunking and SLO-aware step composition.

## Common mistake

Treating it as pure throughput maximization (pack the biggest batch). The real job is admission + preemption under a *hard* KV-memory budget while honoring per-request SLOs: oversubscribe and you thrash on swap/recompute and blow p99; greedily prefill and you starve decodes (TTFT looks fine, TPOT explodes). Fairness and SLO slack, not tokens/sec, drive the policy.

## See also
- [[chunked-prefill]] — the prefill-splitting mechanism the scheduler composes each step
- [[disaggregated-prefill-decode]] — splitting the loop across phase-specialized GPU pools
- [[continuous-batching]] — the iteration-level batching this loop generalizes
