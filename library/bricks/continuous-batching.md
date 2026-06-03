# Continuous Batching

**One-liner:** Iteration-level (token-step) scheduling that admits new requests and retires finished ones at every decode step instead of running a static batch to completion — the throughput backbone of modern LLM serving (vLLM, TGI, TensorRT-LLM).

## The key insight

Static batching pads all requests to the longest sequence and the GPU stalls until the *slowest* request finishes; short requests waste their slots. Continuous batching (a.k.a. iteration-level scheduling, from Orca, OSDI 2022) reschedules the running set every forward pass:

```
batch = {}
while requests remain:
    # admit: pull new requests up to KV-cache budget
    batch ∪= admit(waiting, free_kv_blocks)
    logits = model.step(batch)          # ONE token for every seq
    for r in batch:
        r.append(sample(logits[r]))
        if r.done(EOS / max_len):
            retire(r); free its KV blocks   # slot reused immediately
```

Because prefill and decode have different shapes, schedulers interleave them (chunked prefill / "stall-free" batching) so a long prompt doesn't block decodes.

## Where it appears

- **vLLM** — continuous batching + PagedAttention; admission is gated by free KV *blocks*, not request count
- **HF TGI / TensorRT-LLM (in-flight batching)** — same core loop; NVIDIA calls it "in-flight batching"
- **Orca (OSDI 2022)** — origin of iteration-level scheduling + selective batching
- **Disaggregated / chunked-prefill serving** — continuous batching is the scheduler these systems sit on top of

## Common mistake

Thinking it batches at the request level. It batches at the *token-step* level: at any instant the batch is a heterogeneous mix of requests at different positions, and membership changes every iteration. The binding constraint is KV-cache memory (admission control), not a fixed `max_batch_size`; ignoring this causes preemption/recompute thrash under load.

## See also
- [[continuous-batch-scheduler-admission-control]] — the admission/preemption policy that drives the loop
- [[pagedattention]] — block-level KV allocation that makes per-step admit/retire cheap
- [[chunked-prefill]] — interleaving prefill chunks with decode to keep the batch stall-free
