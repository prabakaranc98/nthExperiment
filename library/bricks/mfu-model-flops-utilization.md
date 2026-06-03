# MFU / Model FLOPs Utilization

**One-liner:** Achieved model-FLOPs throughput as a fraction of hardware peak — MFU = (6ND_throughput) / peak_FLOP/s — the universal scorecard every systems and scaling report quotes to justify parallelism and kernel choices.

## The formula / definition

MFU = (model FLOPs per second) / (peak FLOPs per second of the hardware)

Per-step version (PaLM, Chowdhery et al., 2022):

    model_FLOPs_per_step ≈ 6 · N · B · S
    MFU = (6 · N · B · S / step_time) / (num_chips · peak_FLOP/s_per_chip)

- N = non-embedding parameters, B = batch size (sequences), S = sequence length, so B·S = tokens/step
- 6 = 2 (forward matmul) + 4 (backward) FLOPs per parameter per token
- peak_FLOP/s is dtype-specific: ~990 TFLOP/s BF16 dense on H100, ~1979 TFLOP/s FP8

**MFU vs HFU.** Hardware FLOPs Utilization counts *all* FLOPs the chip executes, including recomputation. MFU counts only the FLOPs the *model definition* requires. Gradient/activation checkpointing inflates HFU above MFU because recompute is real work that does not advance the model. Report MFU for end-to-end efficiency; HFU isolates kernel efficiency.

## Where it appears

- PaLM (2022) — popularized MFU as the headline efficiency metric; reported 46.2% MFU at 540B on TPU v4
- Megatron-LM / MFU studies — used to justify the tensor/pipeline/data parallel split that maximizes achieved FLOPs
- Llama, DeepSeek-V3, MoE training reports — DeepSeek-V3 quotes MFU to argue FP8 training + overlap pushed utilization on H800
- Attention FLOPs caveat: the 6ND approximation ignores attention's O(S²) term, which matters at long context — many reports add 6·N·B·S + 12·L·B·S²·d_model

## Common mistake

Comparing MFU numbers across different peak denominators. A 40% BF16 MFU and a 40% FP8 MFU are not the same throughput — FP8 peak is ~2x higher. Also: quoting MFU without stating whether sparse or dense tensor-core peak was used (the marketing sparse number is 2x dense and is almost never achievable), and confusing it with HFU when checkpointing is on.

## See also
- [[roofline]] — MFU is the system-level summary of whether you are sitting near the compute roof
- [[compute-budget-identity-c-6nd]] — supplies the 6ND numerator MFU divides by peak
- [[gradient-checkpointing]] — the canonical reason MFU and HFU diverge
