# Triton

**One-liner:** A Python-embedded DSL for writing GPU kernels at the *block* level — you express what one program instance does to a tile of data; the compiler handles thread mapping, memory coalescing, shared-memory staging, and instruction scheduling. The default tool for custom fused kernels and the codegen backend of `torch.compile`.

## The programming model

You write one kernel function; it is launched over a **grid** of program instances. Each instance has a `program_id` and operates on a **tile** (block) of the tensors, not individual threads. The unit of work is the block, and pointers/loads/stores are vectorized over `BLOCK_SIZE` (a `tl.constexpr`).

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK + tl.arange(0, BLOCK)   # this instance's tile
    mask   = offs < n                            # guard the ragged tail
    x      = tl.load(x_ptr + offs, mask=mask)    # coalesced HBM -> regs
    y      = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=1024)
```

What the compiler does for you: maps the `BLOCK` to threads/warps, coalesces `tl.load`/`tl.store`, allocates shared memory for `tl.dot` (Tensor-Core matmul) operands, software-pipelines loop stages, and auto-tunes (`@triton.autotune` over `BLOCK`, `num_warps`, `num_stages`). You do **not** write `threadIdx`, `__syncthreads`, or manual smem — that's the abstraction line vs CUDA.

## Where it appears

- **`torch.compile` (Inductor)** — lowers fused pointwise/reduction subgraphs to Triton kernels automatically; this is how most people ship Triton without writing it.
- **FlashAttention** — the official Triton tutorial kernel (and many production attention variants, e.g. in vLLM/xFormers) is written in Triton, not raw CUDA.
- **Liger-Kernel / Unsloth / FlashInfer** — fused RMSNorm, SwiGLU, cross-entropy, RoPE, and decode kernels in Triton for training/inference speedups.
- **Custom quant/MoE ops** — fused dequant-matmul (e.g. GPTQ/AWQ kernels), grouped-GEMM for expert routing, fused optimizer steps.

## Common mistake

Treating Triton like CUDA and reasoning about individual threads. You program at the *block/tile* level — operations are over whole tiles (`tl.arange`, masked load/store, `tl.dot`), and the compiler owns the intra-block thread layout. Two real footguns: forgetting the `mask` on non-multiple-of-block sizes (silent OOB/garbage), and assuming a fused Triton kernel auto-beats cuBLAS for pure GEMM — for large dense matmuls vendor libraries usually still win; Triton's edge is *fusion* of memory-bound chains.

## See also
- [[kernel-fusion]] — Triton is the dominant tool for writing the fused memory-bound kernels
- [[gpu-memory-hierarchy]] — the SRAM/HBM/register structure Triton's tiling abstracts over
- [[flash-attention]] — its reference and many production implementations are Triton kernels
