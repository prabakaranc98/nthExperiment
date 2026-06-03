# Latency-Throughput Tradeoff (TTFT / TPOT / Goodput / SLO)

**One-liner:** LLM serving lives on a Pareto frontier trading per-request latency (TTFT, inter-token latency) against aggregate token throughput; the metric the field now optimizes is **goodput** — throughput counting only requests that meet their latency SLO.

## The definitions

- **TTFT** (Time To First Token) — latency of the *prefill* phase; scales with prompt length and queueing delay. User-perceived "responsiveness."
- **TPOT / ITL** (Time Per Output Token / Inter-Token Latency) — steady-state latency of each *decode* step; the inverse is per-request token speed. User-perceived "generation smoothness."
- **End-to-end latency** ≈ TTFT + (output_len − 1) × TPOT.
- **Throughput** — total tokens/sec (or req/sec) across all concurrent requests; maximized by large batches.
- **Goodput** — throughput counting only requests that satisfy *both* SLOs (e.g. TTFT ≤ 300ms AND TPOT ≤ 50ms). Defined in DistServe (Zhong et al., OSDI 2024).

## The tension

Larger batch → higher throughput (better MFU, amortized weight loads in memory-bound decode) but **higher TPOT** (each step does more work) and **higher TTFT** (head-of-line blocking, prefill contention). The frontier is fundamentally set by the prefill/decode asymmetry: prefill is *compute-bound*, decode is *memory-bound*. Naively co-locating them on the same batch means one phase's batching preference sabotages the other's latency.

## Where it appears

- **Continuous batching** (Orca, vLLM) — iteration-level scheduling raises throughput without proportionally inflating per-request latency by admitting/retiring requests mid-flight.
- **Disaggregated prefill/decode** (DistServe, Splitwise) — runs prefill and decode on separate GPU pools so each is tuned to its own SLO; explicitly optimizes goodput per GPU.
- **Chunked prefill** (Sarathi-Serve, DeepSpeed) — splits long prefills into chunks interleaved with decode to bound TTFT spikes and keep TPOT stable.
- **SLO-aware autoscaling / load balancing** — production stacks (TensorRT-LLM, SGLang) admit and route requests against latency budgets.

## Common mistake

Reporting throughput (tokens/sec) as the headline number with no latency constraint. A system can post huge throughput at a batch size whose TPOT is unusable — those tokens are not goodput. Always quote throughput *at a fixed SLO*, and remember TTFT and TPOT trade off differently, so a single "latency" number is meaningless.

## See also
- [[disaggregated-prefill-decode]] — the architecture built to optimize goodput by splitting the two phases
- [[continuous-batching]] — the scheduler trick that moves the Pareto frontier outward
- [[chunked-prefill]] — bounds TTFT without crushing decode TPOT
