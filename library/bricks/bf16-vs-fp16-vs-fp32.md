# BF16 vs FP16 vs FP32 (Floating-Point Formats)

**One-liner:** Three IEEE-ish float layouts trading mantissa bits (precision) against exponent bits (dynamic range); bf16 keeps fp32's 8-bit exponent so it never overflows, which is why it won large-model training despite having only ~3 decimal digits of precision.

## The formula / definition

A float is `(-1)^s · 1.m · 2^(e - bias)`, partitioned as sign / exponent / mantissa:

| format | bits | exp | mant | bias | max |~rel. eps~| decimal digits |
|--------|------|-----|------|------|-----|------|------|
| fp32   | 32   | 8   | 23   | 127  | ~3.4e38  | 2^-24 ≈ 6e-8 | ~7 |
| fp16   | 16   | 5   | 10   | 15   | 65504    | 2^-11 ≈ 5e-4 | ~3 |
| bf16   | 16   | 8   | 7    | 127  | ~3.4e38  | 2^-8 ≈ 4e-3  | ~2-3 |

Key: **bf16 = fp32 with 16 mantissa bits chopped off.** Same exponent field → same range → truncating fp32→bf16 is just dropping low bits (cheap, no overflow). fp16 has *more* precision than bf16 but tiny range: gradients/activations below ~6e-8 underflow to 0, large values above 65504 overflow to inf.

## Where it appears

- Every frontier pretraining run — weights/activations in bf16, with fp32 master weights + fp32 accumulation (the [[mixed-precision]] recipe). bf16 chosen so loss scaling is unnecessary.
- Pre-2020 fp16 training (V100 era) — required dynamic **loss scaling** (multiply loss by ~2^k before backward, unscale before optimizer) to push gradients out of the fp16 underflow zone.
- TF32 (Ampere tensor cores) — 19-bit internal format: fp32's 8-bit exponent + 10-bit mantissa, the matmul default that motivated bf16's layout.
- Optimizer states — Adam moments often kept fp32 even when weights are bf16 (see [[low-precision-optimizer-states-stochastic]]).

## Common mistake

Believing fp16 is "more accurate than bf16 so it's safer for training." The opposite operationally: fp16's 5-bit exponent makes large-model gradients silently overflow to inf / underflow to 0, causing [[loss-spikes-training-instability]]. bf16 trades precision you can absorb (range matters more than mantissa for LLM dynamics) and accumulates in fp32 anyway. Also: stochastic rounding, not truncation, is what makes low-precision *weight updates* unbiased.

## See also
- [[mixed-precision]] — the master-weights + fp32-accumulate recipe these formats plug into
- [[fp8-low-precision-training]] — pushes the same range-vs-precision tradeoff to 8 bits (E4M3/E5M2)
- [[gpu-memory-hierarchy]] — halving bytes/param is why 16-bit formats exist at all
