# Inference-Time Parallelism (TP / PP / EP for Serving)

**One-liner:** Split weights/experts across GPUs at inference: tensor parallelism (TP) shards every matmul to cut latency, pipeline parallelism (PP) stages layers to add throughput, expert parallelism (EP) shards MoE experts — chosen to fit the model and beat the memory-bandwidth wall that bottlenecks decode.

## The key insight

Decode is memory-bandwidth-bound: each token streams the full weight set from HBM, so per-token latency ≈ (bytes read) / (aggregate HBM bandwidth). Parallelism distributes that read.

- **TP (intra-layer):** shard each weight matrix across G GPUs. Attention splits heads; MLP splits the hidden dim (column-parallel up-proj, row-parallel down-proj). One **all-reduce per layer per token** (2 per transformer block). Cuts both latency and per-GPU memory by ~G, but needs fast NVLink — all-reduce volume = 2·b·s·h per layer, dominates over slow interconnect.
- **PP (inter-layer):** assign layer ranges to stages. Adds aggregate memory and throughput; raises latency via the pipeline bubble. At decode (microbatch of 1 token) the bubble is brutal, so PP is a throughput/capacity tool, not a latency tool.
- **EP (MoE-specific):** place experts on different GPUs; route tokens via **all-to-all dispatch → expert FFN → all-to-all combine**. Lets total expert params exceed one GPU while only top-k experts fire per token. Load imbalance and all-to-all latency are the costs.

Real systems compose them: e.g. TP=8 within a node (NVLink) × PP/EP across nodes (slower fabric). Rule of thumb: TP up to the NVLink domain, then EP/PP/DP beyond it.

## Where it appears

- **DeepSeek-V3 / R1 serving** — large EP (experts spread over many GPUs) with disaggregated prefill/decode; redundant/duplicated hot experts to fix load imbalance.
- **Megatron-LM / TensorRT-LLM / vLLM / SGLang** — TP for low-latency single-replica serving; EP for MoE; PP to fit models that won't fit even sharded by TP.
- **GPT-OSS, Mixtral, Qwen-MoE deployments** — EP is the only way to serve hundreds-of-billions-of-params MoE on a node-bounded HBM budget.

## Common mistake

Treating PP as a latency optimization. PP raises single-request latency (bubble + cross-stage hops) and only helps aggregate throughput by overlapping microbatches. For low latency you want TP (collective per layer, but parallel compute); reach for PP/EP only when the model doesn't fit or you're throughput-bound. Also: pushing TP past the NVLink domain — once all-reduce crosses slow inter-node links, decode latency gets worse, not better.

## See also
- [[tensor-parallel]] — the intra-layer sharding mechanism and its all-reduce cost
- [[expert-parallelism]] — how MoE experts shard and the all-to-all dispatch/combine
- [[prefill-vs-decode]] — why decode is bandwidth-bound and sets the parallelism choice
