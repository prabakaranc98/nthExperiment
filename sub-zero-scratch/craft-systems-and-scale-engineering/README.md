# Craft — Systems & Scale Engineering

**Type:** Craft (Production Function)
**Serves:** Converts framework-brain into ship velocity. The literal research engineering job filter — the difference between someone who understands a paper and someone who can implement it at scale, profile it, and make it fast.

---

## The Problem It Solves

Most ML researchers are framework-users. They know PyTorch's API. They have never written a CUDA kernel. They have never profiled a training run, traced a bottleneck to a memory bandwidth issue, or debugged a distributed training job. They cannot go from a paper to a running, reproducible experiment without scaffolding.

This craft changes that. The goal: be the person who can take any paper from the frontier, implement it from scratch, run it efficiently on GPUs, and make the implementation clean enough that someone else can reproduce it.

---

## What "From Scratch" Means Here
- Implement FlashAttention from scratch in Triton before you use it from a library
- Profile before you optimize. Know *why* something is slow before you fix it
- Every implementation is reproducible: seeded, logged, environment-pinned
- No `torch.compile()` magic without understanding what it's doing

---

## Four Pillars

### 1. Distributed Training
Data parallelism, tensor parallelism, pipeline parallelism — not as concepts but as working code. Understand what happens when a gradient all-reduce fails. Understand what ZeRO-3 is actually doing to memory.

### 2. Triton / Pallas Kernel Writing
Writing custom GPU kernels in Triton (NVIDIA) or Pallas (Google/TPU). The mental model: tiles, shared memory, warps, memory hierarchy. Start with a matrix multiply. Get to FlashAttention.

### 3. Profiling and Debugging
PyTorch Profiler, Nsight Systems, FLOP counting. The discipline: before any optimization, measure. After any optimization, measure again. Never guess.

### 4. Reproducibility Infrastructure
Seeds, experiment tracking (Weights & Biases, or raw), environment pinning (Docker or conda lock files), result verification scripts. The standard: someone else should be able to clone your repo and reproduce your result.

---

## Build Progression

- [ ] **Step 1:** Profile a standard transformer training run. Identify the top 3 bottlenecks. Write down where time is actually spent.

- [ ] **Step 2:** Implement matrix multiplication in Triton. Benchmark against `torch.matmul`. Match within 10%.

- [ ] **Step 3:** Implement a fused attention kernel in Triton (softmax + attention in one pass). Compare to naive PyTorch.

- [ ] **Step 4:** Run a 2-GPU data-parallel training job using `torchrun`. Monitor GPU utilization, communication overhead. Identify where the scaling efficiency drops.

- [ ] **Step 5:** Implement ZeRO Stage 1 manually (partition optimizer states across devices). Verify memory savings match the theory.

- [ ] **Step 6:** Reproduce one paper from the frontier (full training pipeline, not just inference). Make it reproducible: README with exact commands, seeded, environment pinned.

---

## Essential Resources

- [ ] **Triton documentation + tutorials** — triton-lang.org
  - Tutorial 1: Vector addition
  - Tutorial 2: Fused softmax
  - Tutorial 3: Matrix multiplication

- [ ] **CUDA Programming Guide (selected)**
  - Ch. 1–2: Thread hierarchy, memory model
  - Ch. 5: Performance guidelines

- [ ] **FlashAttention paper — Dao et al. (2022)**
  - §2: Background (tiling, IO complexity)
  - §3: FlashAttention algorithm — implement this

- [ ] **ZeRO paper — Rajbhandari et al. (2019)**
  - §3: ZeRO Stage 1, 2, 3 — understand each stage's memory savings

- [ ] **Megatron-LM — Shoeybi et al. (2019)**
  - §3: Tensor model parallelism — implement a 2-layer version

- [ ] **PyTorch Profiler documentation**

- [ ] **Making Deep Learning Go Brrrr — Horace He** *(blog post)*
  - Full post — the mental model for ML performance optimization

---

## Build Log

| # | What I Built | Benchmark vs. Baseline | Reproducible? | Date |
|---|-------------|----------------------|---------------|------|
| — | — | — | — | — |
