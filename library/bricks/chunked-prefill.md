# Chunked Prefill

**One-liner:** Split a long prompt's prefill into fixed-size token chunks and interleave them with ongoing decode steps in the same batch, capping the latency spike a big prefill imposes by trading TTFT for TPOT.

## The key insight

A full prefill of P tokens is one giant compute-bound GEMM; decode steps are tiny memory-bound GEMVs. Run alone, a long prefill monopolizes a forward pass and stalls every other request's decode (TPOT spike / generation "stutter"). Instead, cap prefill work per step at a token budget B (the chunk size, e.g. 512). Each scheduler step assembles a batch up to B tokens: it greedily fills with decode tokens (1 per running request) first, then packs prefill chunks of the remaining budget — fusing the two phases into one mixed forward pass.

```
budget B   # max tokens per scheduled step (e.g. 512)
each step:
  tokens = [1 decode token per running seq]      # protect TPOT
  rem    = B - len(tokens)
  while rem > 0 and waiting prefills exist:
    take min(rem, remaining_prompt) tokens as a prefill chunk
    rem -= chunk
  one forward pass over tokens (varlen attention; chunk attends to its KV prefix)
```

Smaller B -> lower, smoother TPOT but higher TTFT and more attention recompute over re-read KV; larger B -> faster TTFT, lumpier decode. Each chunk still attends to all previously-prefilled KV, so total prefill FLOPs are unchanged; only the scheduling granularity changes.

## Where it appears

- Sarathi / Sarathi-Serve (2023-24) — origin of "chunked-prefills + stall-free batching"; coined the TTFT/TPOT framing and the decode-prioritized token budget.
- vLLM — `enable_chunked_prefill` with `max_num_batched_tokens` as B; default-on for long contexts to bound decode latency.
- TensorRT-LLM, SGLang, DeepSpeed-FastGen ("dynamic SplitFuse") — same fuse-prefill-with-decode scheduling under different names.

## Common mistake

Treating it as a substitute for prefill/decode disaggregation. Chunking shares one engine and still couples the phases on the same GPUs; disaggregation puts them on separate hardware. Also: setting B too small inflates attention cost (each chunk re-reads the growing KV prefix) and tanks TTFT for negligible TPOT gain.

## See also
- [[prefill-vs-decode]] — the two-phase distinction chunking exploits and blends
- [[continuous-batching]] — the iteration-level scheduler chunked prefill plugs into
- [[disaggregated-prefill-decode]] — the alternative: separate the phases instead of fusing them
