# Cosine LR Decay

**One-liner:** Anneal the learning rate along a half-cosine curve from peak to a small floor over a pre-committed token/step horizon — the Chinchilla-era compute-optimal default that WSD now challenges.

## The formula / definition

After linear warmup to η_max over T_warm steps, for step t in [T_warm, T]:

η(t) = η_min + 0.5·(η_max − η_min)·(1 + cos(π · (t − T_warm) / (T − T_warm)))

The horizon T must be fixed in advance: the cosine is steepest in the middle and flattens near both ends, so η starts at η_max, passes through (η_max+η_min)/2 at the midpoint, and lands on the floor η_min (often η_max/10, sometimes 0) exactly at t = T.

## Where it appears

- **Chinchilla (Hoffmann et al., 2022)** — established that the cosine cycle length should match the *full* training run; setting T longer than the actual run leaves loss on the table, and the compute-optimal scaling-law fits assume a matched cosine.
- **GPT-3, Llama 1/2, most pre-2024 pretraining** — peak → ~10% floor cosine is the canonical schedule.
- **WSD / WSD-style schedules (MiniCPM, DeepSeek, OLMo-2)** — the explicit baseline cosine is compared against; WSD holds LR constant then decays only at the end, matching or beating cosine while allowing continued training from any checkpoint.

## Common mistake

Decoupling the cosine period from the actual stopping point. Because T is baked into η(t), you cannot cleanly extend training past T (LR is already at the floor) and a too-long T means you never reach the floor — both hurt final loss. This horizon-locking is exactly the rigidity WSD was designed to remove.

## See also
- [[warmup-stable-decay-schedule]] — the horizon-decoupled successor cosine is benchmarked against
- [[learning-rate-warmup]] — the linear ramp that precedes the cosine descent
- [[scaling-laws]] — Chinchilla compute-optimal fits assume a matched-length cosine
