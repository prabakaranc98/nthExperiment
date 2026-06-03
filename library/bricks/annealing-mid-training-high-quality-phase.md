# Annealing / Mid-Training High-Quality Phase

**One-liner:** A short final pretraining phase (~last few % of tokens) that upweights the highest-quality and benchmark-relevant data (code, math, instructions, textbooks) while the LR decays to ~0, cheaply buying large benchmark gains; near-universal in Llama 3, OLMo, MiniCPM recipes.

## The key insight

Decouple data quality from the schedule's "memorability." During the high-LR stable phase, weight updates are large and later data overwrites earlier — so the *final* tokens seen as LR → 0 disproportionately shape the model. Concentrate scarce premium data there.

Mechanically it is the decay arm of a Warmup-Stable-Decay (WSD) schedule fused with a data-mixture switch:

```
phase 1 (stable, ~95-99% of tokens): LR = η_max, bulk web mixture
phase 2 (anneal,  ~1-5%  of tokens): LR: η_max → ~0 (linear/cosine/1-sqrt)
                                      mixture → upweight code/math/QA/instruct
```

Equivalent framings: "mid-training," "data annealing," WSD decay, μ-decay. Because LR is already low, you can branch *multiple* annealing runs from one stable checkpoint to A/B test data mixes — annealing is the standard probe for measuring a data source's marginal value (OLMo, DataDecide).

## Where it appears

- **OLMo / OLMo 2** — explicit two-stage: stable on Dolma, then anneal on Dolmino (curated math/code/instruction) with LR→0; large MMLU/GSM8K jumps.
- **Llama 3** — final "annealing" stage upweights high-quality sources and uses annealing on small domain sets to *evaluate* data value before committing.
- **MiniCPM** — introduced WSD; the decay phase is where high-quality + SFT-style data is injected, enabling cheap model "families" from one trunk.
- **Phi / Nemotron / DeepSeek** — synthetic, textbook, and reasoning data concentrated in late/mid-training stages.

## Common mistake

Treating annealing as just an LR-schedule choice. The benchmark lift comes mainly from the *data mixture shift*; annealing on the same web mixture gains little. Conversely, dumping benchmark-adjacent data here without decontamination is how contamination sneaks in — the late, high-influence position amplifies leakage.

## See also
- [[warmup-stable-decay-schedule]] — the LR schedule whose decay arm *is* the annealing window
- [[data-mixing-laws]] — predicts the optimal upweighting of domains during the phase
- [[model-based-quality-filtering]] — how the "high-quality" subset for annealing is selected
