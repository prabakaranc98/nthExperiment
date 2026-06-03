# 2:4 Semi-Structured Sparsity

**One-liner:** Force exactly 2 of every 4 contiguous weights to zero so NVIDIA's Sparse Tensor Cores skip the zeros and deliver ~2x matmul throughput — the only sparsity pattern with broad hardware acceleration (Ampere+).

## The formula / definition

Partition each row of a weight matrix into contiguous groups of 4. In each group keep the 2 largest-magnitude entries, zero the other 2:

  for each 4-tuple (w0,w1,w2,w3): keep the 2 with largest |w|, set rest = 0

Storage: 50% nonzeros packed as compressed values + a 2-bit-per-element metadata index encoding which 2 of 4 survived. The hardware feeds the dense activation through the metadata to select the matching operands, halving the effective inner-product length.

  Dense GEMM: A (M×K) · W (K×N)
  2:4 GEMM:  same M,N,K logically, but W stored at K/2 effective columns/group → ~2x FLOP/s on Sparse Tensor Cores

Crucially it is *structured at fine grain*: the 2-of-4 constraint matches the Tensor Core's MMA tiling, unlike free unstructured sparsity which gives memory savings but no speedup on GPUs.

## Where it appears

- NVIDIA Ampere/Hopper/Blackwell Sparse Tensor Cores — the cuSPARSELt / `torch.sparse.to_sparse_semi_structured` path that turns 50% sparsity into real wall-clock speedup
- SparseGPT & Wanda — one-shot pruning methods that produce 2:4 masks for LLMs without retraining, recovering accuracy via per-output-channel calibration / weight-update
- ASP (Automatic SParsity) and "prune → fine-tune (often with permutation) → recompress" recipes used to ship 2:4 LLaMA/BERT checkpoints
- Combined with INT8/FP8 in inference stacks to stack the ~2x sparse speedup on top of low-bit quantization

## Common mistake

Expecting the ~2x speedup just from having 50% zeros. Random/unstructured 50% sparsity gets ~0x speedup on GPUs — you must satisfy the *exact* 2-of-4-per-contiguous-group layout (plus channel permutation to align), and you only see the gain on Sparse-Tensor-Core kernels for the supported dtypes; small or memory-bound GEMMs may not benefit at all.

## See also
- [[sparsegpt-wanda-one-shot-pruning]] — the standard one-shot methods for finding accurate 2:4 masks in LLMs
- [[structured-vs-unstructured-pruning]] — 2:4 is the fine-grained structured pattern that bridges accuracy and hardware speedup
- [[tensor-tile-mma-cores]] — the MMA tiling that 2:4 metadata must align with to get acceleration
