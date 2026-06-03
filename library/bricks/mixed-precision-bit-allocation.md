# Mixed-Precision Bit Allocation

**One-liner:** Assign different bitwidths per layer/channel/group by a sensitivity score (Hessian-trace or gradient-based), keeping the few sensitive components at high precision and aggressively quantizing the rest to hit a target average bit budget.

## The formula / definition

Frame it as constrained optimization: choose per-block bitwidth b_i to minimize total quantization-induced loss increase subject to an average-bit (or memory) budget.

  min_{b}  Σ_i  Ω_i(b_i)     s.t.   Σ_i size_i · b_i  ≤  B_target

where Ω_i is the sensitivity of block i. Second-order (HAWQ) estimate of the loss perturbation from quantizing weights W_i to Ŵ_i:

  Ω_i  ≈  (1/n) · Tr(H_i) · ‖Ŵ_i − W_i‖²₂      (H_i = Hessian block; Tr(H_i) via Hutchinson)

Higher Tr(H_i) ⇒ flatter quantization tolerance is needed ⇒ allocate more bits. Cheaper proxies: gradient magnitude / Fisher diagonal (g²) in place of Tr(H), or activation-magnitude saliency (AWQ-style). The integer program is solved by sorting blocks by Ω_i / size_i, by a Pareto frontier, or by an ILP/greedy knapsack to pick the bit assignment {2,3,4,8,…} per block.

## Where it appears

- **HAWQ / HAWQ-V2 / HAWQ-V3** — Hessian-trace sensitivity assigns mixed INT bitwidths per layer; V3 produces hardware-deployable integer-only mixed-precision schedules via ILP.
- **LLM.int8() / mixed FP16+INT8** — route the ~0.1% outlier feature dimensions through FP16, quantize the rest to INT8 (per-column precision split).
- **SqueezeLLM / SpQR / extreme sub-4-bit** — keep a sparse set of sensitive (outlier) weights in FP16, store the dense majority at 3-4 bit; an empirical bit-allocation by saliency.
- **MXFP / microscaling block formats** — per-block shared scales let bit budget effectively vary across the tensor on Blackwell-class hardware.

## Common mistake

Confusing this with mixed-precision *training* (FP16/BF16 compute + FP32 master weights). That is a uniform numerical-format choice for stability; bit allocation is a *per-component, sensitivity-driven* assignment of differing bitwidths to minimize accuracy loss under a budget. Also: treating uniform-bit average as equivalent — two schemes at the same mean bitwidth can differ wildly in loss if the sensitive blocks were under-allocated.

## See also
- [[quantization]] — bit allocation is the layer/channel-granularity policy layered on top of the base quant scheme
- [[extreme-sub-4-bit-quantization]] — relies on keeping a sensitive minority high-precision while pushing the rest below 4 bits
- [[fisher-information-natural-gradient]] — the Fisher/Hessian diagonal is the sensitivity score driving allocation
