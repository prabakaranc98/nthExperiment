# Pretraining Data Curriculum

**One-liner:** Deliberately ordering or phasing the training data — quality/difficulty ramps, domain phase-ins, source up-weighting — instead of i.i.d. sampling from a fixed mixture, most commonly concentrating the highest-quality and reasoning-heavy data in the final WSD decay phase.

## The key insight

Loss on a token depends on *when* it is seen, not just *whether* it is seen, because the LR schedule modulates how strongly late gradients reshape weights. Curriculum exploits two facts: (1) data seen during the low-LR decay phase is "locked in" and disproportionately shapes the final model; (2) the optimal data *mixture* is non-stationary — broad web text early builds general capability, then high-quality/instruction/math/code data late steers the endpoint.

A typical phased schedule with the WSD (warmup-stable-decay) LR:

```
Phase 1 (warmup + stable, high LR, ~70-90% of tokens):
    sample ~ broad web mixture (Common Crawl, dedup'd, light filtering)
Phase 2 (decay, LR → 0, final ~10-30% of tokens):
    sample ~ high-quality mixture, up-weight {curated web, math,
             code, textbooks, synthetic rephrase, instruction-like}
```

Equivalently the per-step mixture weight w_d(t) over domains d is a function of training step t (a *schedule*), not a constant. Difficulty curricula instead order *examples* by a difficulty score (perplexity, length, model-judged quality) low → high.

## Where it appears

- **MiniCPM / WSD annealing** — explicitly concentrate high-quality + SFT-style data in the decay phase; quality of decay-phase data dominates final benchmarks.
- **Llama 3, DeepSeek-V3, Nemotron, OLMo 2** — multi-phase pretraining with a distinct long-context + high-quality "mid-training"/annealing stage at the tail.
- **Phi series** — "textbooks-are-all-you-need": curriculum *is* the curation (synthetic, filtered-for-educational-value data ordering).
- **Datamix / domain-reweighting work (DoReMi, data-mixing laws)** — provides the per-phase weights a curriculum schedules over time.

## Common mistake

Treating it as "save the good data for last" without controlling repetition and LR. Putting all high-quality data only in a short, very-low-LR decay window means the model barely updates on it (small effective LR → small weight change), and over-concentrating a narrow domain at the tail causes forgetting of earlier general capability. The decay-phase mixture and its LR magnitude must be co-designed.

## See also
- [[annealing-mid-training-high-quality-phase]] — the decay/mid-training phase this curriculum targets
- [[warmup-stable-decay-schedule]] — the LR schedule whose decay window the curriculum exploits
- [[data-mixing-laws]] — predicts the per-phase domain weights a curriculum sequences
