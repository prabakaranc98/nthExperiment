# Compute Budget Identity C ≈ 6ND

**One-liner:** Total training FLOPs are roughly six times non-embedding parameters times training tokens — the back-of-envelope behind every token budget, GPU-hour estimate, and scaling-law fit.

## The formula / definition

C ≈ 6 · N · D

- N = non-embedding parameters
- D = training tokens
- C = total training FLOPs (forward + backward)

**Where the 6 comes from.** A parameter in a matmul costs ~2 FLOPs per token in the forward pass (one multiply + one add). Backward is ~2x forward (gradient w.r.t. inputs + gradient w.r.t. weights). So 2 (fwd) + 4 (bwd) = 6 FLOPs per parameter per token. Inference-only is the forward half: **C_infer ≈ 2ND**.

**Per-token forward cost** ≈ 2N, so a single forward over D tokens is 2ND; training adds the 4ND backward term.

## Where it appears

- **Kaplan 2020 / Chinchilla (Hoffmann 2022)** — the substitution that converts a (N, D) loss surface into a single compute axis C, so you can solve for compute-optimal N\* ∝ C^0.5, D\* ∝ C^0.5.
- **GPU-hour / wall-clock estimates** — divide C by (cluster FLOP/s × MFU) to get training time; e.g. Llama-3-70B: 6·70e9·15e12 ≈ 6.3e24 FLOPs.
- **Inference cost modeling** — 2ND per generated token (× KV-cache overhead) drives the over-training-for-serving decision.

## Common mistake

Forgetting that the 6ND identity counts **only the dense matmul FLOPs and ignores attention's O(L²·d) term**. At long context the QK^T and attention-value products are not captured by 6ND; the correction is roughly +6 · n_layers · L · d per token, which becomes non-negligible past ~10k tokens. Also: for MoE use **active** params (per-token), not total params, in N.

## See also
- [[scaling-laws]] — uses C = 6ND to collapse the loss surface onto a compute axis
- [[mfu-model-flops-utilization]] — turns the C estimate into real wall-clock via achieved FLOP/s
- [[inference-optimal-over-training]] — trades training C for lower 2ND serving cost
