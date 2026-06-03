# Warmup-Stable-Decay (WSD) Schedule

**One-liner:** Three-phase LR schedule — linear warmup, a long constant plateau at peak LR, then a short fast decay — that decouples the schedule from a fixed total step count, so the plateau checkpoint is reusable and training is continuable for any compute budget.

## The schedule

Given peak LR η, warmup steps W, total steps T, and decay fraction f (typically 0.1–0.2):

```
η(t) = η · t/W                       for t < W           (warmup)
η(t) = η                             for W ≤ t < (1−f)T  (stable plateau)
η(t) = η · g((t − (1−f)T)/(fT))      for t ≥ (1−f)T      (decay)
```

where g goes 1 → ~0 over the decay window (linear, cosine, or 1/√ / exp; MiniCPM found exp-like decay strong). The key property: the stable phase has **no dependence on T**. The loss stays on a high plateau during the plateau, then drops sharply during decay — the "WSD cliff."

## Why it matters

- The plateau checkpoint is a **reusable base**: branch from it and decay to any target token count without re-running warmup+plateau. Enables a Chinchilla-style scaling-law sweep from a single trunk run.
- Decoupling LR from T removes the cosine-schedule requirement to commit to a final step count up front — naturally continuable / compute-agnostic.

## Where it appears

- **MiniCPM (Hu et al., 2024)** — origin of the WSD name; pairs the decay phase with a high-quality data mix and shows the cliff matches/beats cosine at the same compute.
- **DeepSeek-V2/V3, MiniCPM** — multi-step (staircase) variants and the decay phase double as the annealing / high-quality mid-training window.
- **Continual pretraining & "infinite LR" schedules** — re-warm from a plateau checkpoint to extend training without restart.

## Common mistake

Comparing a *plateau-phase* WSD checkpoint to a cosine end-of-run checkpoint and concluding WSD is worse. WSD's loss is supposed to sit high until the decay — you must evaluate (and harvest) the **post-decay** checkpoint. Also: skipping decay entirely, or making f too small, loses most of the final loss drop.

## See also
- [[cosine-lr-decay]] — the schedule WSD replaces; cosine bakes the total step count T into η(t)
- [[annealing-mid-training-high-quality-phase]] — the decay phase is where the high-quality data mix is injected
- [[scaling-laws]] — one plateau trunk + many short decays cheaply traces the loss-vs-tokens curve
