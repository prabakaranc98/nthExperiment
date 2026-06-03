# Tensor / Tile-MMA Cores

**One-liner:** Specialized matrix-multiply-accumulate units (NVIDIA Tensor Cores) that deliver the bulk of a GPU's FLOP/s for GEMMs; on Hopper/Blackwell they are driven by an async warp-group (wgmma) + TMA programming model that decouples data movement from compute.

## The primitive

A Tensor Core computes a small tiled MMA in one instruction: **D = A·B + C**, with A,B,C,D tiles of fixed shape, accumulating in higher precision than the inputs.

- Inputs in FP16/BF16/FP8/FP4/INT8 (per generation); **accumulate in FP32** (or INT32). This split-precision accumulation is why low-bit inputs don't blow up.
- Instruction evolution: Volta `mma` (per-warp, 16×16×16) → Ampere async copy → **Hopper `wgmma`** (warp-group, 128 threads, one issuing thread group drives the whole tile, async) → **Blackwell `tcgen05` / 5th-gen** Tensor Cores with native FP4 (NVFP4/MXFP4) and per-block microscaling.

## The Hopper/Blackwell programming model

The throughput is gated by feeding the cores, not by the MMA itself. Two async engines hide latency:

- **TMA (Tensor Memory Accelerator):** a hardware DMA engine that copies multi-dim tiles HBM↔SMEM with a single descriptor, computing addresses/bounds in hardware. Frees the SMs from address math.
- **wgmma:** asynchronous warp-group MMA — issue the matmul, then `wgmma.commit`/`wait` later, overlapping with the next TMA load. Combined with a software pipeline (mbarrier-synchronized SMEM stages) this gives the **producer/consumer warp-specialization** that CUTLASS 3.x and FlashAttention-3 rely on.

Goal: keep Tensor Cores ~100% busy while TMA streams the next tiles → approach peak FLOP/s and high **MFU**.

## Where it appears

- **FlashAttention-3** — warp-specialized wgmma + TMA pipeline; overlaps softmax (non-MMA) with GEMM, ships FP8 attention on H100.
- **CUTLASS 3.x / CuTe** — exposes wgmma/TMA as composable "collective" GEMM kernels; the de-facto template for custom mixed-precision matmuls.
- **DeepSeek-V3 / FP8 training** — fine-grained FP8 GEMMs with FP32 accumulation on Hopper Tensor Cores; a frontier-scale validation of low-bit tensor-core training.
- **Blackwell FP4 inference** — NVFP4/MXFP4 microscaling formats run directly on 5th-gen Tensor Cores, ~2× FP8 throughput.

## Common mistake

Treating Tensor Core peak TFLOP/s as achievable by just calling a matmul. The MMA units are rarely the bottleneck — **feeding them is.** Without TMA + async pipelining (or tiny/odd tile shapes, unaligned strides, memory-bound shapes), you stall on HBM/SMEM and hit a fraction of peak. Also: the inputs are low-precision but the **accumulator is FP32** — don't assume the whole GEMM is done in FP16/FP8.

## See also
- [[roofline]] — Tensor Cores raise the peak-FLOP/s ceiling; only helps when compute-bound
- [[fp8-low-precision-training]] — the low-bit input formats Hopper/Blackwell cores consume with FP32 accumulate
- [[mfu-model-flops-utilization]] — measures how close you get to Tensor Core peak in practice
