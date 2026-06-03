# CUDA Graphs & Kernel-Launch Overhead

**One-liner:** Capture a fixed sequence of GPU kernels once into a replayable graph, then relaunch the whole graph with a single API call — eliminating the per-kernel CPU launch latency that dominates memory-bound, small-batch decode.

## The key insight

Each `cudaLaunchKernel` costs ~5–10 µs of CPU-side overhead (driver dispatch, arg marshalling, queue submission). One LLM decode step launches hundreds of tiny kernels (norms, matmuls, attention, rotary, sampling). When each kernel runs in only a few µs, the GPU sits idle waiting on the CPU to enqueue the next one — the step is **CPU-launch-bound**, not compute- or bandwidth-bound.

CUDA Graphs collapse this. Record the launch sequence once into a DAG of operations, then replay:

```
# Capture (once, per static shape)
g = CUDAGraph()
with capture(stream):          # records launches, does NOT execute
    model.decode_step(static_in)   # writes into pre-allocated static buffers
# Replay (every step)
copy_(static_in, new_token)    # mutate the captured input buffers in place
g.replay()                     # ONE driver call dispatches the whole DAG
out = static_out.clone()
```

Cost model: launch overhead drops from O(#kernels · t_launch) to O(1) per step. For decode, the relevant inequality is `#kernels · t_launch ≳ t_step_gpu` — i.e. graphs help exactly when launch time rivals the GPU work, which is the small-batch / short-kernel regime.

## Where it appears

- **vLLM / SGLang / TensorRT-LLM** — capture decode-step graphs per (batch size, seq bucket); replay each iteration. Major TPOT win at low batch. vLLM keeps an "eager mode" fallback and pre-captures a set of bucketed batch sizes at startup.
- **PyTorch `torch.compile` (CUDA Graph trees) & `make_graphed_callables`** — graph the compiled forward to remove Python + launch overhead; the `mode="reduce-overhead"` path.
- **NVIDIA Megatron / training** — graph repeated micro-batch steps; pairs with kernel fusion (fewer, bigger kernels → graphs matter less but still help dispatch).
- **Mamba / SSM decode** — many tiny scan/elementwise kernels per step make launch overhead acute; graphs are near-mandatory for competitive latency.

## Common mistake

Treating a captured graph as dynamic. A graph hard-codes pointers, shapes, and control flow — **everything must be static**. New inputs must be written *into the same pre-allocated buffers* (you replay, you don't re-pass args). Changing batch size, sequence length, or any data-dependent branch invalidates the graph; that's why serving stacks capture a discrete set of shape buckets and pad to the nearest one. Capturing under dynamic allocation, or with CPU↔GPU syncs / `.item()` / host-side conditionals in the loop, silently breaks capture or replays stale work.

## See also
- [[kernel-fusion]] — the complementary lever: fewer kernels per step means less to launch and less to graph
- [[inference-and-serving]] — graphs are a standard decode-path optimization in continuous-batching servers
- [[prefill-vs-decode]] — graphs target the launch-bound decode phase, not compute-bound prefill
